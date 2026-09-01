"""
Configuration settings for Flower Plant Health Recommendation System.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
STATIC_FOLDER = BASE_DIR / "static"
SAMPLE_IMAGES_FOLDER = STATIC_FOLDER / "sample_images"
MODEL_WEIGHTS_DIR = BASE_DIR / "ml_model" / "weights"
MODEL_PATH = MODEL_WEIGHTS_DIR / "plant_health_model.pth"
KNOWLEDGE_BASE_PATH = BASE_DIR / "recommendation" / "knowledge_base.json"

# Ensure runtime directories exist
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
SAMPLE_IMAGES_FOLDER.mkdir(parents=True, exist_ok=True)
MODEL_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (224, 224)
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max

# Supported Flower Plant Species and Diseases
CLASS_NAMES = [
    "Rose___Healthy",
    "Rose___Black_Spot",
    "Rose___Powdery_Mildew",
    "Rose___Aphid_Infestation",
    "Rose___Rust",
    "Hibiscus___Healthy",
    "Hibiscus___Leaf_Blight",
    "Hibiscus___Aphid_Infestation",
    "Hibiscus___Iron_Chlorosis",
    "Marigold___Healthy",
    "Marigold___Alternaria_Blight",
    "Marigold___Powdery_Mildew",
    "Marigold___Spider_Mites",
    "Jasmine___Healthy",
    "Jasmine___Leaf_Rust",
    "Jasmine___Bud_Borer_Damage",
    "Jasmine___Nutrient_Deficiency",
    "General_Flower___Healthy",
    "General_Flower___Fungal_Leaf_Spot",
    "General_Flower___Overwatering_Stress",
    "General_Flower___Dehydration_Stress"
]

NUM_CLASSES = len(CLASS_NAMES)
