import os
import time
import json
import uuid
import cv2
import io   # <-- Added for PDF BytesIO
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from flask import Flask, request, jsonify, render_template, session, redirect, url_for, send_file
from flask_cors import CORS
from torchvision import transforms
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from model import HybridCNNViT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.units import inch
import requests 

# ---------- Load environment ----------
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
GOOGLE_PLACES_KEY = os.getenv("GOOGLE_PLACES_KEY")  # put your Google Places API key in .env

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found — please set it in your .env file")
if not GOOGLE_PLACES_KEY:
    # Allow server start but return error from nearby_doctors if missing
    print("Warning: GOOGLE_PLACES_KEY not set in .env — /nearby_doctors will fail until set.")


# ---------- Hugging Face Zephyr client ----------
client = InferenceClient(
    model="HuggingFaceH4/zephyr-7b-alpha",
    token=HF_TOKEN
)

# ---------- Setup ----------
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app)
app.secret_key = "ryaniscool007"

last_report_data = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load adult model
model_adult = HybridCNNViT().to(device)
model_adult.load_state_dict(torch.load("model_checkpoint_epoch10.pth", map_location=device))
model_adult.eval()

# Load pediatric model
model_pediatric = HybridCNNViT().to(device)
model_pediatric.load_state_dict(torch.load("p_model_checkpoint_epoch10.pth", map_location=device))
model_pediatric.eval()

class_labels = ['Normal', 'Pneumonia']

# ---------- CLAHE ----------
class CLAHETransform:
    def _call_(self, img):
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl_img = clahe.apply(img_cv)
        return Image.fromarray(cl_img).convert('RGB')

transform = transforms.Compose([
    CLAHETransform(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------- Image validation ----------
def is_chest_xray(image: Image.Image) -> bool:
    img_np = np.array(image)
    if len(img_np.shape) == 2:
        gray = img_np
    elif len(img_np.shape) == 3 and img_np.shape[2] == 3:
        r, g, b = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2]
        diff_rg = np.mean(np.abs(r - g))
        diff_rb = np.mean(np.abs(r - b))
        diff_gb = np.mean(np.abs(g - b))
        if diff_rg < 15 and diff_rb < 15 and diff_gb < 15:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            return False
    else:
        return False

    h, w = gray.shape
    aspect_ratio = h / w
    if aspect_ratio < 0.6 or aspect_ratio > 2.0:
        return False

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist_norm = hist / hist.sum()
    dark_pixels = np.sum(hist_norm[:20])
    bright_pixels = np.sum(hist_norm[180:])
    if dark_pixels < 0.02 or bright_pixels < 0.01:
        return False
    if h < 200 or w < 200:
        return False
    return True

# ---------- Grad-CAM ----------
def generate_gradcam(chosen_model, image_tensor, class_idx, output_path):
    image_tensor = image_tensor.to(device)
    chosen_model.eval()
    gradients = []
    activations = []

    def backward_hook(chosen_module, grad_input, grad_output):
        gradients.append(grad_output[0])
    def forward_hook(chosen_module, input, output):
        activations.append(output)

    target_layer = chosen_model.cnn_features[-1]
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    output = chosen_model(image_tensor)
    chosen_model.zero_grad()
    class_score = output[0, class_idx]
    class_score.backward()

    grads = gradients[0].detach().cpu()
    acts = activations[0].detach().cpu()
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * acts).sum(dim=1).squeeze()
    cam = torch.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam_np = cam.numpy()

    orig_img = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    orig_img = np.clip(orig_img * [0.229, 0.224, 0.225] + 
                       [0.485, 0.456, 0.406], 0, 1)
    cam_resized = cv2.resize(cam_np, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(np.uint8(orig_img * 255), 0.5, heatmap, 0.5, 0)

    gradcam_filename = f"{uuid.uuid4().hex}_gradcam.jpg"
    gradcam_path = os.path.join(UPLOAD_FOLDER, gradcam_filename)
    cv2.imwrite(gradcam_path, overlay)

    forward_handle.remove()
    backward_handle.remove()
    return gradcam_path, cam_np

# ---------- ViT Attention ----------
def extract_vit_attention(chosen_model, img_tensor, save_path='static/vit_attention.png'):
    vit = chosen_model.vit
    with torch.no_grad():
        outputs = vit(pixel_values=img_tensor.unsqueeze(0), output_attentions=True)
        attn = outputs.attentions[-1] if outputs.attentions else None
    if attn is None:
        return None

    attn_map = attn[0, :, 0, 1:]
    attn_map = attn_map.mean(0).reshape(14, 14).cpu().numpy()
    attn_map = cv2.resize(attn_map, (224, 224))
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())

    img_np = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * np.array([0.229, 0.224, 0.225]) +
              np.array([0.485, 0.456, 0.406]))
    img_np = np.clip(img_np, 0, 1)

    heatmap = cv2.applyColorMap(np.uint8(255 * attn_map), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    overlay = 0.4 * heatmap + 0.6 * img_np
    plt.imsave(save_path, overlay)
    return save_path

# ---------- Hotspot summarization ----------
def summarize_cam_quadrants(cam_np):
    try:
        h, w = cam_np.shape
        mid_h, mid_w = h // 2, w // 2
        q1 = cam_np[0:mid_h, 0:mid_w].mean()
        q2 = cam_np[0:mid_h, mid_w:w].mean()
        q3 = cam_np[mid_h:h, 0:mid_w].mean()
        q4 = cam_np[mid_h:h, mid_w:w].mean()
        quads = [q1, q2, q3, q4]
        names = ["upper-left", "upper-right", "lower-left", "lower-right"]
        thresh = 0.40 * max(quads)
        hotspots = [names[i] for i, val in enumerate(quads) if val >= thresh]
        if not hotspots:
            sorted_idx = sorted(range(4), key=lambda i: quads[i], reverse=True)
            hotspots = [names[sorted_idx[0]]]
            if quads[sorted_idx[1]] > 0.6 * quads[sorted_idx[0]]:
                hotspots.append(names[sorted_idx[1]])
        return hotspots
    except Exception:
        return []

# ---------- Prompts ----------
def build_patient_prompt(prediction, confidence, age, hotspots, triage):
    hotspots_text = ", ".join(hotspots) if hotspots else "no clear localized hotspot"
    return (
        f"You are a empathetic,compassionate medical assistant. Based on the facts below, "
        f"write a clear, concise, patient-friendly health guidance with these sections:\n\n"
        f"🔹 Overview: A 1-2 sentence summary of the model result in simple language.\n"
        f"🔹 Recommended Next Steps: Bullet points prioritizing urgent actions first.\n"
        f"🔹 Preventive Tips: (only if result is normal or mild)\n"
        f"🔹 Important Note: Include a professional disclaimer reminding to consult a doctor.\n\n"
        f"Facts:\n"
        f"- Model result: {prediction} (confidence {confidence:.2%})\n"
        f"- Age: {age}\n"
        f"- Heatmap hotspots: {hotspots_text}\n"
        f"- Triage level: {triage}\n\n"
        f"Use plain language, avoid jargon, and be empathetic."
f"""
Your job:
1. Only talk about findings directly related to pneumonia or lung health. Do not mention unrelated conditions (e.g., cholesterol, cancer, heart disease) unless explicitly given in the input.
2. Output structured patient-friendly advice with these sections:

🔹 Overview: One or two sentences summarizing result in friendly, clear language.  
   - If "Normal" → sound relieved and reassuring, celebrate the good news, and optionally suggest routine health habits.  
   - If "Pneumonia" or abnormal → explain simply and calmly what this means.
   
🔹 Recommended Next Steps:  
   - If "Normal" → Suggest routine checkups, healthy habits, and when to seek care if symptoms arise.  
   - If "Pneumonia"or abnormal → Provide urgent-but-calm steps: when to see a doctor, possible next tests, symptom watchlist.

🔹 Preventive Tips:  
   - If "Normal" → Mention lung health tips (hydration, exercise, avoid smoking, vaccines).  
   - If "Pneumonia" or abnormal → Mention recovery tips and prevention of worsening.

🔹 Important Note:  
   Always include: "This is AI-generated preliminary guidance. Please consult a qualified medical professional for a final diagnosis."

Tone guide:
- For "Normal" → Warm, cheerful, slightly celebratory but still cautious.  
- For "Pneumonia" or abnormal → Calm, supportive, reassuring, but clear about urgency.
- Avoid jargon. Use short sentences.

Output in Markdown format so it can be rendered in HTML with icons and bold text.
"""
    )

    

def build_clinician_prompt(prediction, confidence, age, hotspots, triage):
    hotspots_text = ", ".join(hotspots) if hotspots else "no clear localized hotspot"
    return (
        f"You are a technical assistant for clinicians. Provide a concise interpretation (2-4 sentences) "
        f"of the heatmaps and model result, including:\n"
        f"- Likely lung zones involved (e.g., lower lobe consolidation)\n"
        f"- Whether Grad-CAM and ViT attention maps agree\n"
        f"- Whether the pattern suggests localized consolidation, diffuse interstitial changes, or is uncertain\n\n"
        f"Facts:\n"
        f"- Model result: {prediction} (confidence {confidence:.2%})\n"
        f"- Age: {age}\n"
        f"- Heatmap hotspots: {hotspots_text}\n"
        f"- Triage level: {triage}\n\n"
        f"Keep it factual and end with 'Recommend radiologic and clinical correlation.'"
       f"""

Your job:
1. Only talk about findings directly related to pneumonia or lung health. Do not mention unrelated conditions (e.g., cholesterol, cancer, heart disease) unless explicitly given in the input.
2. Output structured patient-friendly advice with these sections:

🔹 Overview: One or two sentences summarizing result in friendly, clear language.  
   - If "Normal" → sound relieved and reassuring, celebrate the good news, and optionally suggest routine health habits.  
   - If "Pneumonia" or abnormal → explain simply and calmly what this means.
   
🔹 Recommended Next Steps:  
   - If "Normal" → Suggest routine checkups, healthy habits, and when to seek care if symptoms arise.  
   - If "Pneumonia" or abnormal → Provide urgent-but-calm steps: when to see a doctor, possible next tests, symptom watchlist.

🔹 Preventive Tips:  
   - If "Normal" → Mention lung health tips (hydration, exercise, avoid smoking, vaccines).  
   - If "Pneumonia" or abnormal → Mention recovery tips and prevention of worsening.

🔹 Important Note:  
   Always include: "This is AI-generated preliminary guidance. Please consult a qualified medical professional for a final diagnosis."

Tone guide:
- For "Normal" → Warm, cheerful, slightly celebratory but still cautious.  
- For "Pneumonia"or abnormal → Calm, supportive, reassuring, but clear about urgency.
- Avoid jargon. Use short sentences.

Output in Markdown format so it can be rendered in HTML with icons and bold text.
"""
    )


# ---------- Routes ----------
@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/awareness',methods=['GET'])
def awareness():
    return render_template('awareness.html')

@app.route('/questionnaire', methods=['GET'])
def questionnaire_form():
    return render_template('form.html')

@app.route('/submit_questionnaire', methods=['POST'])
def submit_questionnaire():
    fever=request.form.get('fever')
    cough_severity=request.form.getlist('cough_severity')
    chest_pain=request.form.get('chest_pain')
    breathing=request.form.getlist('breathing')
    sleep_hours=request.form.getlist('sleep_hours')
    water_intake=request.form.getlist('water_intake')
    medications=request.form.get('medications')
    physical_activity=request.form.getlist('physical_activity')
    smoking_status = request.form.get('smoking_status')
    symptoms = request.form.getlist('symptoms')

    session['questionnaire'] = {
        'fever':fever,
        'cough_severity':cough_severity,
        'chest_pain':chest_pain,
        'breathing':breathing,
        'sleep_hours':sleep_hours,
        'water_intake':water_intake,
        'medications':medications,
        'physical_activity':physical_activity,
        'smoking_status': smoking_status,
        'symptoms': symptoms,

    }

    # After saving questionnaire, redirect to X-ray upload page
    return redirect(url_for('index'))


@app.route('/diagnose')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global last_report_data
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        file = request.files['image']
        age = int(request.form.get("age", 0))
        gender = request.form.get("gender", "N/A")
        name = request.form.get("name", "Anonymous")
        img = Image.open(file).convert('RGB')
        if not is_chest_xray(img):
            return jsonify({'error': 'Uploaded image is not recognized as a chest X-ray'}), 400

        img_tensor = transform(img).to(device)

        if age <= 18:
            chosen_model = model_pediatric
        else:
            chosen_model = model_adult    

        with torch.no_grad():
            outputs = chosen_model(img_tensor.unsqueeze(0))
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, pred_class = torch.max(probs, 1)
            pred_class = pred_class.item()
            confidence = confidence.item()
            prediction = class_labels[pred_class] if pred_class < len(class_labels) else "Unknown"
            conf_score = f"{(100*confidence):.2f}"

        gradcam_path, cam_np = generate_gradcam(chosen_model, img_tensor.unsqueeze(0), pred_class, "static/gradcam.png")
        attention_path = extract_vit_attention(chosen_model, img_tensor, "static/vit_attention.png")
        hotspots = summarize_cam_quadrants(cam_np)
        triage = "urgent" if prediction == "Pneumonia" and confidence > 0.85 else "routine"

        # -------- LLM Guidance --------
        def call_zephyr(prompt):
            try:
                completion = client.chat_completion(
                    messages=[
                        {"role": "system", "content": "You are a concise, helpful medical assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500
                )
                return completion.choices[0].message["content"].strip()
            except Exception as e:
                print(f"Zephyr API error: {e}")
                return "Unable to fetch AI guidance."

        llm_guidance = call_zephyr(build_patient_prompt(prediction, confidence, age, hotspots, triage))
        llm_explanation = call_zephyr(build_clinician_prompt(prediction, confidence, age, hotspots, triage))

        # -------- Save for PDF --------
        last_report_data = {
            "name": request.form.get("name", "Anonymous"),
            "age": request.form.get("age", "N/A"),
            "gender": request.form.get("gender", "N/A"),
            "label": prediction,
            "confidence": conf_score,
            "gradcam_url": gradcam_path,
            "attention_url": attention_path,
            "guidance": llm_guidance,            # <-- Added
            "explanation": llm_explanation, 
            "questionnaire":
        session.get('questionnaire', {})
        }

        return jsonify({
            'label': prediction,
            'confidence': conf_score,
            'gradcam_url': gradcam_path,
            'attention_url': attention_path,
            'health_guidance': llm_guidance,
            'heatmap_explanation': llm_explanation
        })

    except Exception as e:
        print(f"Prediction route error: {e}")
        return jsonify({'error': 'Prediction failed due to server error'}), 500

        #from flask import request, jsonify

#GOOGLE_PLACES_KEY = "AiLL9DPaL9WAZ-NeFu2i_zUEAn3SUIzaSyCRfn3"
   #(Google Places) ----------
@app.route("/nearby_doctors", methods=["GET"])
def nearby_doctors():
    lat = request.args.get("lat")
    lon = request.args.get("lon")

    if not lat or not lon:
        return jsonify({"error": "Missing coordinates lat and lon"}), 400

    # ===== 1️⃣ Try Google Places API =====
    if GOOGLE_PLACES_KEY:
        url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        params = {
            "location": f"{lat},{lon}",
            "radius": 5000,  # 5 km
            "type": "doctor",
            "key": GOOGLE_PLACES_KEY
        }
        try:
            resp = requests.get(url, params=params, timeout=8)
            data = resp.json()
            print("Google Places API raw response:", data)

            # If Google works & returns results
            if data.get("status") == "OK" and data.get("results"):
                return jsonify(data)
        except Exception as e:
            print(f"Google Places API error: {e}")

    # ===== 2️⃣ Fallback: OpenStreetMap Nominatim =====
    try:
        osm_url = "https://nominatim.openstreetmap.org/search"
        osm_params = {
            "q": "doctor",
            "format": "json",
            "limit": 10,
            "lat": lat,
            "lon": lon
        }
        headers = {"User-Agent": "DeepLungsAI/1.0"}
        osm_resp = requests.get(osm_url, params=osm_params, headers=headers, timeout=8)
        osm_data = osm_resp.json()

        # Format OSM results to match Google-like structure
        formatted_results = [
            {
                "name": place.get("display_name"),
                "geometry": {
                    "location": {
                        "lat": float(place["lat"]),
                        "lng": float(place["lon"])
                    }
                }
            }
            for place in osm_data
        ]

        return jsonify({"results": formatted_results, "status": "OK"})
    except Exception as e:
        return jsonify({"error": f"Both Google and OSM failed: {e}"}), 500
@app.route("/air_quality", methods=["GET"])
def air_quality():
    lat = request.args.get("lat")
    lon = request.args.get("lon")

    if not lat or not lon:
        return jsonify({"error": "Missing coordinates lat and lon"}), 400

    if not GOOGLE_PLACES_KEY:
        return jsonify({
            "currentConditions": {
                "indexes": [{"value": 75}],
                "pollutants": [{"code": "PM2.5"}]
            }
        })

    try:
        url = f"https://airquality.googleapis.com/v1/currentConditions:lookup?key={GOOGLE_PLACES_KEY}"
        payload = {
            "location": {"latitude": float(lat), "longitude": float(lon)},
            "extraComputations": ["HEALTH_RECOMMENDATIONS"],
            "languageCode": "en"
        }

        resp = requests.post(url, json=payload, timeout=8)
        resp.raise_for_status()
        raw = resp.json()

        # Extract AQI & pollutant
        idx_value = None
        pollutant_code = None
        if "indexes" in raw and raw["indexes"]:
            idx_value = raw["indexes"][0].get("aqi")
            pollutant_code = raw["indexes"][0].get("dominantPollutant")

        return jsonify({
            "currentConditions": {
                "indexes": [{"value": idx_value}],
                "pollutants": [{"code": pollutant_code}]
            }
        })

    except requests.RequestException as e:
        print(f"[ERROR] AQI API request failed: {e}")
        return jsonify({"error": "Failed to fetch air quality"}), 500



from flask import send_file
import io
import os
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from PIL import Image, ImageDraw

# Function to create a circular logo
def make_circle_image(image_path, size=(60, 60)):
    img = Image.open(image_path).convert("RGBA")
    img = img.resize(size, Image.LANCZOS)

    # Create circular mask
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, size[0], size[1]), fill=255)
    img.putalpha(mask)

    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    return img_bytes

@app.route('/download_report', methods=['GET'])
def download_report():
    global last_report_data
    if not last_report_data:
        return "No report data available. Please run a diagnosis first.", 400

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )
    styles = getSampleStyleSheet()
    story = []

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=20,
        spaceAfter=20,
        alignment=0  # left align
    )
    heading_style = ParagraphStyle(
        'HeadingCustom',
        parent=styles['Heading2'],
        spaceAfter=10,
        textColor=colors.HexColor("#0B5394")
    )

    # ---- Logo + Title Row ----
    logo_path = os.path.join("static", "images", "deep_lungs_logo.png")
    if os.path.exists(logo_path):
        circular_logo = make_circle_image(logo_path)
        logo = RLImage(circular_logo, width=0.8*inch, height=0.8*inch)
        title = Paragraph(" DeepLungs AI Diagnosis Report", title_style)

        table_data = [[logo, title]]
        table = Table(table_data, colWidths=[1*inch, 5*inch])
        table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
            ('TOPPADDING', (0, 0), (-1, -1), 0),
        ]))
        story.append(table)
    else:
        story.append(Paragraph("🫁 DeepLungs AI Diagnosis Report", title_style))

    story.append(Spacer(1, 20))

    # ---- Patient Info ----
    patient_data = [
        ["Name", last_report_data['name']],
        ["Age", str(last_report_data['age'])],
        ["Gender", last_report_data['gender']]
    ]
    patient_table = Table(patient_data, colWidths=[1.5*inch, 4.5*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#E1EAF2")),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 15))

    # ---- Diagnosis ----
    story.append(Paragraph(f"<b>Diagnosis:</b> {last_report_data['label']}", styles['Normal']))
    story.append(Paragraph(f"<b>Confidence:</b> {last_report_data['confidence']}", styles['Normal']))
    story.append(Spacer(1, 15))

    # ---- Guidance ----
    story.append(Paragraph("AI Health Guidance", heading_style))
    story.append(Paragraph(last_report_data['guidance'].replace("\n", "<br/>"), styles['Normal']))
    story.append(Spacer(1, 15))

    # ---- Questionnaire ----
    if last_report_data.get("questionnaire"):
        story.append(Paragraph("Patient Questionnaire Responses", heading_style))
        for key, value in last_report_data["questionnaire"].items():
            if isinstance(value, list):
                value = ", ".join(value) if value else "N/A"
            story.append(Paragraph(f"<b>{key.replace('_', ' ').title()}:</b> {value}", styles['Normal']))
        story.append(Spacer(1, 15))

    # ---- Images ----
    for img_path, label in [
        (last_report_data['gradcam_url'], "Grad-CAM Heatmap"),
        (last_report_data['attention_url'], "ViT Attention Map")
    ]:
        if img_path and os.path.exists(img_path):
            story.append(Paragraph(label, heading_style))
            story.append(RLImage(img_path, width=4*inch, height=4*inch))
            story.append(Spacer(1, 12))

    # ---- Build PDF ----
    doc.build(story)
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"DeepLungs_Report_{last_report_data['name'].replace(' ', '_')}.pdf",
        mimetype='application/pdf'
    )

if __name__ == '_main_':
    app.run(debug=True)