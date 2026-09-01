import os
import logging
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

from config import (
    UPLOAD_FOLDER,
    SAMPLE_IMAGES_FOLDER,
    ALLOWED_EXTENSIONS,
    MAX_CONTENT_LENGTH,
    CLASS_NAMES
)
from ml_model.model import PlantHealthInferenceEngine
from ml_model.preprocessor import validate_image, load_and_preprocess_image
from recommendation.engine import RecommendationEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

logger.info("Initializing ML inference engine and recommendation engine...")
inference_engine = PlantHealthInferenceEngine()
recommendation_engine = RecommendationEngine()
logger.info("Engines loaded successfully.")

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "supported_classes_count": len(CLASS_NAMES),
        "model_loaded": inference_engine.model_loaded
    })

@app.route("/api/sample-images", methods=["GET"])
def get_sample_images():
    samples = [
        {
            "id": "rose_black_spot",
            "name": "Rose with Black Spot",
            "plant": "Rose",
            "condition": "Black Spot Disease",
            "filename": "rose_black_spot.jpg",
            "expected_class": "Rose___Black_Spot"
        },
        {
            "id": "rose_healthy",
            "name": "Healthy Rose",
            "plant": "Rose",
            "condition": "Healthy Foliage",
            "filename": "rose_healthy.jpg",
            "expected_class": "Rose___Healthy"
        },
        {
            "id": "rose_powdery_mildew",
            "name": "Rose Powdery Mildew",
            "plant": "Rose",
            "condition": "Powdery Mildew",
            "filename": "rose_powdery_mildew.jpg",
            "expected_class": "Rose___Powdery_Mildew"
        },
        {
            "id": "rose_aphids",
            "name": "Rose Aphids",
            "plant": "Rose",
            "condition": "Aphid Infestation",
            "filename": "rose_aphids.jpg",
            "expected_class": "Rose___Aphid_Infestation"
        },
        {
            "id": "hibiscus_healthy",
            "name": "Healthy Hibiscus",
            "plant": "Hibiscus",
            "condition": "Healthy Foliage",
            "filename": "hibiscus_healthy.jpg",
            "expected_class": "Hibiscus___Healthy"
        },
        {
            "id": "hibiscus_blight",
            "name": "Hibiscus Leaf Blight",
            "plant": "Hibiscus",
            "condition": "Fungal Blight",
            "filename": "hibiscus_blight.jpg",
            "expected_class": "Hibiscus___Leaf_Blight"
        },
        {
            "id": "hibiscus_chlorosis",
            "name": "Hibiscus Iron Chlorosis",
            "plant": "Hibiscus",
            "condition": "Nutrient Chlorosis",
            "filename": "hibiscus_chlorosis.jpg",
            "expected_class": "Hibiscus___Iron_Chlorosis"
        },
        {
            "id": "marigold_healthy",
            "name": "Healthy Marigold",
            "plant": "Marigold",
            "condition": "Healthy Specimen",
            "filename": "marigold_healthy.jpg",
            "expected_class": "Marigold___Healthy"
        },
        {
            "id": "marigold_blight",
            "name": "Marigold Alternaria Blight",
            "plant": "Marigold",
            "condition": "Alternaria Leaf Blight",
            "filename": "marigold_blight.jpg",
            "expected_class": "Marigold___Alternaria_Blight"
        },
        {
            "id": "marigold_spider_mites",
            "name": "Marigold Spider Mites",
            "plant": "Marigold",
            "condition": "Spider Mite Damage",
            "filename": "marigold_spider_mites.jpg",
            "expected_class": "Marigold___Spider_Mites"
        },
        {
            "id": "jasmine_healthy",
            "name": "Healthy Jasmine",
            "plant": "Jasmine",
            "condition": "Healthy Specimen",
            "filename": "jasmine_healthy.jpg",
            "expected_class": "Jasmine___Healthy"
        },
        {
            "id": "jasmine_rust",
            "name": "Jasmine Leaf Rust",
            "plant": "Jasmine",
            "condition": "Leaf Rust Infection",
            "filename": "jasmine_rust.jpg",
            "expected_class": "Jasmine___Leaf_Rust"
        },
        {
            "id": "flower_overwatering",
            "name": "Overwatering Stress",
            "plant": "General Flower",
            "condition": "Overwatering Root Stress",
            "filename": "flower_overwatering.jpg",
            "expected_class": "General_Flower___Overwatering_Stress"
        },
        {
            "id": "flower_dehydration",
            "name": "Dehydration Stress",
            "plant": "General Flower",
            "condition": "Underwatering / Dehydration",
            "filename": "flower_dehydration.jpg",
            "expected_class": "General_Flower___Dehydration_Stress"
        }
    ]
    return jsonify({"samples": samples})

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        sample_filename = request.form.get("sample_filename")
        
        if sample_filename:
            image_path = SAMPLE_IMAGES_FOLDER / secure_filename(sample_filename)
            if not image_path.exists():
                return jsonify({"success": False, "error": "Sample image not found"}), 404
            image_source = image_path
            image_url = f"/static/sample_images/{sample_filename}"
        elif "image" in request.files:
            file = request.files["image"]
            if file.filename == "":
                return jsonify({"success": False, "error": "No file selected"}), 400
            if not allowed_file(file.filename):
                return jsonify({"success": False, "error": "Unsupported file extension. Allowed: PNG, JPG, JPEG, WEBP, BMP"}), 400
            
            filename = secure_filename(file.filename)
            save_path = UPLOAD_FOLDER / filename
            file.save(save_path)
            image_source = save_path
            image_url = f"/uploads/{filename}"
        else:
            return jsonify({"success": False, "error": "No image payload provided"}), 400

        _, pil_img = load_and_preprocess_image(image_source)
        is_valid, validation_msg = validate_image(pil_img)
        if not is_valid:
            return jsonify({"success": False, "error": validation_msg}), 400

        pred_result = inference_engine.predict(image_source, top_k=3)
        top_class = pred_result["top_class"]
        confidence = pred_result["confidence"]
        top_predictions = pred_result["all_predictions"]

        recommendation = recommendation_engine.get_recommendation(
            class_name=top_class,
            confidence=confidence,
            top_predictions=top_predictions
        )

        return jsonify({
            "success": True,
            "image_url": image_url,
            "prediction": pred_result,
            "recommendation": recommendation
        })

    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        return jsonify({"success": False, "error": f"Diagnostic processing failed: {str(e)}"}), 500

@app.route("/api/ask-doctor", methods=["POST"])
def ask_doctor():
    try:
        data = request.get_json() or {}
        question = data.get("question", "")
        plant = data.get("plant", "")
        condition = data.get("condition", "")

        if not question:
            return jsonify({"success": False, "error": "Question text is required"}), 400

        answer = recommendation_engine.ask_doctor(question, current_plant=plant, current_condition=condition)
        return jsonify({"success": True, "answer": answer})
    except Exception as e:
        logger.error(f"Ask Doctor error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, secure_filename(filename))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting Flower Plant Health Recommendation System on port {port}...")
    app.run(host="127.0.0.1", port=port, debug=True)
