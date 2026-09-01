# 🌸 Flower Plant Health Recommendation System (FloraCare AI)

An AI/ML-powered end-to-end web system designed to identify flowering plants, detect and classify their health conditions (fungal infections, pest infestations, nutrient chlorosis, overwatering, dehydration, and healthy states), and generate structured, actionable, and personalized plant care recommendations.

---

## 🌟 Key Highlights & Differentiators

Unlike standard disease detection systems that only output a raw disease label, this system provides a **comprehensive care strategy** with:
1. **Decision Support & Overlap Disclaimers**: Clarifies visual overlaps (e.g. distinguishing overwatering vs iron chlorosis vs fungal root rot).
2. **Immediate Triage Actions**: 24-48 hour intervention steps to stop disease spread.
3. **Structured Care Profiles**: Tailored protocols for Watering, Sunlight, Soil/pH, Organic Treatments, Chemical Alternatives, and Long-Term Prevention.
4. **Interactive Care Checklist**: Step-by-step interactive checkboxes to track gardening tasks.
5. **Printable / PDF Care Sheets**: Export formatted treatment plans for field or greenhouse use.
6. **Ask Plant Doctor AI**: Integrated chatbot for follow-up questions and symptom inquiries.

---

## 🏗️ Architecture

```
User / Gardener
      │
      ▼ (Upload / Camera Snap / Sample)
Frontend (HTML5 / CSS3 / Vanilla JS)
      │
      ▼ (Multipart POST /api/predict)
Flask Backend (app.py)
      │
      ├──> Image Preprocessor (Resize, Normalize, Transforms)
      │
      ├──> MobileNetV2 Transfer Learning CNN (ml_model/model.py)
      │      └──> Softmax Probabilities & Top-k Differential Predictions
      │
      ├──> Recommendation Engine (recommendation/engine.py)
      │      └──> Cross-references Structured Knowledge Base (knowledge_base.json)
      │
      ▼
Personalized Care Plan + Diagnostic Report
```

---

## 🌺 Supported Plants and Diseases

| Plant Species | Conditions Detected |
|---|---|
| **Rose (*Rosa*)** | Healthy Foliage, Black Spot (*Diplocarpon rosae*), Powdery Mildew, Aphids (*Macrosiphum rosae*), Rose Rust (*Phragmidium*) |
| **Hibiscus (*Hibiscus rosa-sinensis*)** | Healthy Foliage, Leaf Blight (*Alternaria/Cercospora*), Aphids/Mealybugs, Iron Chlorosis (*Nutrient deficiency*) |
| **Marigold (*Tagetes*)** | Healthy Foliage, Alternaria Leaf Blight, Powdery Mildew, Spider Mites (*Tetranychus urticae*) |
| **Jasmine (*Jasminum*)** | Healthy Foliage, Leaf & Twig Rust (*Uromyces hobsonii*), Bud Borer Damage, Nutrient Deficiency |
| **General Flowering Plants** | Healthy Specimen, Fungal Leaf Spot, Overwatering & Hypoxia Stress, Underwatering / Dehydration |

---

## 🚀 How to Run the Application

### 1. Requirements
- Python 3.10+
- PyTorch & Torchvision
- Flask
- Pillow, NumPy, Requests

### 2. Quick Start
Double-click `run.bat` or run:
```bash
python app.py
```
Then open your browser at **http://127.0.0.1:5000**.

### 3. Running Unit Tests
```bash
python test_pipeline.py
```

---

## 📡 REST API Endpoints

- `GET /api/health`: Health status and model load verification.
- `GET /api/sample-images`: List of preloaded test gallery samples.
- `POST /api/predict`: Image diagnosis endpoint (supports multipart file `image` or `sample_filename`).
- `POST /api/ask-doctor`: Symptom & botanical Q&A endpoint.
