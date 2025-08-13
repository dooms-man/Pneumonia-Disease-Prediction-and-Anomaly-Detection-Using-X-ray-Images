# 🩺 DeepLungs AI — Hybrid CNN + Vision Transformer for Pneumonia Detection

DeepLungs AI is an advanced AI-driven diagnostic system designed to detect pneumonia from chest X-ray images.  
Our approach uses a **Hybrid CNN (DenseNet121) + Vision Transformer (ViT)** architecture to combine the strengths of convolutional and attention-based models for superior accuracy and explainability.

---

## 📑 Index  
1. [Model Architecture](#-model-architecture)
2. [Features](#-Features)
3. [Image Preprocessing](#️-image-preprocessing)  
4. [Results](#-results)  
5. [Research References](#-research-references)  
6. [Team](#-team)  

---

## 🧠 Model Architecture

### 1️⃣ CNN Feature Extractor
- **Base**: DenseNet121 (pretrained on ImageNet).
- Outputs **1024 feature channels** after global average pooling.

### 2️⃣ Vision Transformer (ViT)
- **Model**: `google/vit-base-patch16-224` (pretrained).
- Extracts **768-dimensional embeddings** from image patches.

### 🚀 Features

- **Dual Specialized Models** — Separate CNN+ViT hybrid models for **adult** and **pediatric** patients for age-specific accuracy.  
- **Advanced Image Preprocessing** — CLAHE-based enhancement for better lung detail visibility.  
- **Automated Chest X-ray Validation** — Rejects non-X-ray images before running predictions.  
- **Explainable AI**  
  - **Grad-CAM Heatmaps** — Highlights pneumonia-affected lung regions.  
  - **ViT Attention Maps** — Visualizes model’s attention focus.  
  - **Hotspot Summarization** — Detects and names affected lung quadrants.  
- **AI-Generated Medical Guidance** — Hugging Face Zephyr LLM produces:  
  - **Patient-friendly summaries** with next steps & prevention tips.  
  - **Clinician-oriented interpretations** for medical decision support.  
- **Interactive Patient Questionnaire** — Collects key symptom and lifestyle data to personalize results.  
- **Doctor Finder** — Locate nearby healthcare providers using Google Places API or OpenStreetMap fallback.  
- **Air Quality Integration** — Fetches live AQI data with health recommendations.  
- **One-Click PDF Reports** — Beautifully formatted report with:  
  - Patient info  
  - AI predictions & confidence  
  - Health guidance  
  - Questionnaire responses  
  - Grad-CAM & ViT attention images  
  - Professional disclaimer

---

### 🖼️ Image Preprocessing
- **CLAHE (Contrast Limited Adaptive Histogram Equalization)**  
  - `clipLimit=2.0`, `tileGridSize=(8,8)` to enhance contrast in X-ray images.
- **Resize**: Images resized to `(224, 224)` for model input compatibility.
- **Augmentations** (Training Only):  
  - Random horizontal flip (simulate variations in patient posture)  
  - Random rotation (±10°) to improve generalization

---

### 📊 Results

### **Adult Model (Hybrid CNN + ViT)**
| Class        | Precision | Recall | F1-Score |
|--------------|-----------|--------|----------|
| Normal (0)   | 1.00      | 1.00   | 1.00     |
| Pneumonia (1)| 1.00      | 1.00   | 1.00     |
**Accuracy**: **99.82%** 

---

### **Pediatric Model (Hybrid CNN + ViT)**
| Class        | Precision | Recall | F1-Score |
|--------------|-----------|--------|----------|
| Normal (0)   | 0.99      | 0.97   | 0.98     |
| Pneumonia (1)| 0.98      | 0.99   | 0.99     |
**Accuracy**: **98.8%** (Test set: Pediatric X-rays)

---

### 📚 Research References
This project builds upon insights from:
1. Dosovitskiy et al., *"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"* — [ViT Paper](https://arxiv.org/abs/2010.11929)
2. Huang et al., *"Densely Connected Convolutional Networks"* — [DenseNet Paper](https://arxiv.org/abs/1608.06993)
3. Raghu et al., *"Do Vision Transformers See Like Convolutional Neural Networks?"* — [ViT vs CNN Paper](https://arxiv.org/abs/2108.08810)
4. Generative Deep Learning by David Foster

---

### 👨‍⚕️ Team
**DeepLungs AI Hackathon Team** — Innovating medical diagnostics with AI 🚀
Team Members: Akshara,Aditya,Manas,Manryan