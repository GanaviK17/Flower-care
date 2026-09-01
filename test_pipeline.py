import os
import io
import json
import unittest
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(r"C:\Users\Ganavi K\.gemini\antigravity\scratch\flower_plant_health_system")
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from config import CLASS_NAMES, NUM_CLASSES, KNOWLEDGE_BASE_PATH
from ml_model.preprocessor import load_and_preprocess_image, validate_image
from ml_model.model import PlantHealthInferenceEngine
from recommendation.engine import RecommendationEngine
from app import app

class TestPlantHealthSystem(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.rec_engine = RecommendationEngine()
        self.inference_engine = PlantHealthInferenceEngine()

    def test_knowledge_base_completeness(self):
        """Verify that every supported class in CLASS_NAMES has a rich knowledge base profile."""
        with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
            kb = json.load(f)
        
        for cls in CLASS_NAMES:
            self.assertIn(cls, kb, f"Class {cls} missing from knowledge base!")
            profile = kb[cls]
            self.assertTrue(len(profile["species"]) > 0)
            self.assertTrue(len(profile["condition_name"]) > 0)
            self.assertTrue(len(profile["description"]) > 0)
            self.assertTrue(len(profile["visual_symptoms"]) > 0)
            self.assertTrue(len(profile["immediate_actions"]) > 0)
            self.assertIn("watering", profile)
            self.assertIn("sunlight_environment", profile)
            self.assertIn("soil_nutrition", profile)
            self.assertIn("treatments", profile)
            self.assertIn("prevention", profile)
            self.assertIn("checklist_tasks", profile)

    def test_preprocessor(self):
        """Verify image loader, tensor sizing, and validation."""
        img = Image.new("RGB", (300, 300), color=(50, 150, 50))
        tensor, pil_img = load_and_preprocess_image(img)
        self.assertEqual(tensor.shape, (1, 3, 224, 224))
        
        is_valid, msg = validate_image(pil_img)
        self.assertTrue(is_valid)

    def test_recommendation_engine(self):
        """Verify that recommendation engine generates proper output structure."""
        rec = self.rec_engine.get_recommendation(
            "Rose___Black_Spot",
            confidence=0.89,
            top_predictions=[
                {"plant": "Rose", "condition": "Black Spot", "confidence": 0.89, "confidence_percent": 89.0},
                {"plant": "Rose", "condition": "Healthy", "confidence": 0.08, "confidence_percent": 8.0}
            ]
        )
        self.assertEqual(rec["species"], "Rose (Rosa)")
        self.assertEqual(rec["confidence_percentage"], 89.0)
        self.assertEqual(rec["reliability"], "High Confidence")
        self.assertTrue(len(rec["immediate_actions"]) > 0)
        self.assertTrue(len(rec["checklist_tasks"]) > 0)

    def test_ask_doctor(self):
        """Verify Ask Doctor AI responses for botanical queries."""
        ans = self.rec_engine.ask_doctor("Why are my rose leaves turning yellow?")
        self.assertIn("Yellow Leaves Diagnosis Guide", ans)
        
        ans_pest = self.rec_engine.ask_doctor("How do I get rid of aphids?")
        self.assertIn("Pest Control Protocol", ans_pest)

    def test_api_health(self):
        """Verify /api/health endpoint."""
        res = self.app.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["supported_classes_count"], NUM_CLASSES)

    def test_api_sample_images(self):
        """Verify /api/sample-images endpoint."""
        res = self.app.get("/api/sample-images")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("samples", data)
        self.assertTrue(len(data["samples"]) >= 10)

    def test_api_predict_sample(self):
        """Verify /api/predict with a sample image filename."""
        res = self.app.post("/api/predict", data={"sample_filename": "rose_black_spot.jpg"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertIn("prediction", data)
        self.assertIn("recommendation", data)
        self.assertIn("checklist_tasks", data["recommendation"])

if __name__ == "__main__":
    unittest.main()
