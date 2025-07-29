# Pneumonia-Disease-Prediction-and-Anomaly-Detection-Using-X-ray-Images
Hybrid Pneumonia Detection Project (ResNet-50 + ViT-B16 + Age Feature)
=======================================================================

1. Project Overview

Goal: Develop a hybrid deep learning model that detects Pneumonia using both chest X-ray images and patient age.
- Model Type: Hybrid CNN + Transformer model.
- Architecture: ResNet-50 (CNN) for local image features + ViT-B16 (Vision Transformer) for global image features.
- Additional Feature: Patient age integrated into the model to improve diagnostic accuracy.

2. Dataset Details



3. Data Preprocessing

- Resize images to 224x224 pixels.
- Normalize pixel values to range [0, 1].
- Handle missing or invalid ages; normalize age (0-120) to [0, 1].
- Balance datasets using oversampling and augmentation (rotation, flipping, etc.).

4. Model Architecture

- Two parallel input branches:
  - ResNet-50 CNN (pretrained on ImageNet, backbone frozen initially).
  - ViT-B16 Transformer (pretrained, handles global context).
- Age input branch: simple dense layers to process numeric age.
- Feature fusion: concatenation of CNN, ViT, and age features.
- Classifier head: Dense layers + sigmoid activation for binary classification.

5. Training Pipeline

- Loss: Binary Cross-Entropy.
- Optimizer: Adam with learning rate scheduling.
- Metrics: Accuracy, Precision, Recall, AUC.
- Hardware: Trained on GPUs (T4 or A100); CPU possible but slower.
- Training Duration: Multiple epochs until validation accuracy stabilizes.

6. Frontend Design

- Tech Stack: HTML5, CSS3, JavaScript.
- Features:
    - Age input (number field).
    - X-ray image upload (file input, JPEG/PNG).
    - Image preview before submission.
    - Analyze button sends data to backend API.
    - Displays prediction (Pneumonia/Normal) and confidence score.
- Responsive design for desktop and mobile.

7. Backend API
-Framework: Flask (Python).
- Endpoint: POST /predict
- Accepts: multipart form data (X-ray image + age).
- Preprocessing: resizes and normalizes image, normalizes age.
- Feeds inputs to trained model and returns prediction as JSON:
    {
      "prediction": "Pneumonia",
    
    }

8. Deployment Plan

- Local testing: Flask backend + frontend on localhost (http://127.0.0.1:5000).
- Production: Deploy backend on cloud (Heroku, AWS, Render) and host frontend (Netlify, GitHub Pages).
- Ensure secure API communication (HTTPS) and CORS handling.

9. Project Roadmap

- Collect and clean datasets (adult + pediatric).
- Build and train hybrid model.
- Develop backend API for inference.
- Design frontend web interface.
- Integrate age + X-ray upload and connect frontend to backend.
- Deploy complete system online.
