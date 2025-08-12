# 🩺 DeepLungs AI — Hybrid CNN + Vision Transformer for Pneumonia Detection

DeepLungs AI is an advanced AI-driven diagnostic system designed to detect pneumonia from chest X-ray images.  
Our approach uses a **Hybrid CNN (DenseNet121) + Vision Transformer (ViT)** architecture to combine the strengths of convolutional and attention-based models for superior accuracy and explainability.

---

## ✨ Key Features
- **Dual Models**: Separate models for **Adults** and **Pediatrics** to ensure dataset-specific optimization.
- **Hybrid Architecture**: CNN for local spatial feature extraction + ViT for global context understanding.
- **CLAHE Preprocessing**: Enhances contrast in medical images to improve feature visibility.
- **Data Augmentation**: Random flips, rotations for better generalization.
- **Grad-CAM Explainability**: Visualizes areas influencing the model’s predictions.

---

## 🧠 Model Architecture

### 1️⃣ CNN Feature Extractor
- **Base**: DenseNet121 (pretrained on ImageNet).
- Outputs **1024 feature channels** after global average pooling.

### 2️⃣ Vision Transformer (ViT)
- **Model**: `google/vit-base-patch16-224` (pretrained).
- Extracts **768-dimensional embeddings** from image patches.

### 3️⃣ Feature Fusion + Classifier
```plaintext
[ CNN Features (1024) ] + [ ViT Features (768) ]
             ↓ Concatenate
      Fully Connected Layer (512 → 2 classes)
- **CNN Branch**: DenseNet121 extracts spatial & texture features from X-ray images.
- **ViT Branch**: Vision Transformer captures global dependencies and context.
- **Fusion**: Concatenate DenseNet output (1024 features) with ViT output (768 features).
- **Classifier**: Fully Connected (512 → 2 neurons) with ReLU + Dropout for binary classification.

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

---

### 👨‍⚕️ Team
**DeepLungs AI Hackathon Team** — Innovating medical diagnostics with AI 🚀
Team Members: Akshara,Aditya,Manas,Manryan