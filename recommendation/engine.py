"""
Recommendation Engine for Flower Plant Health System.
Bridges ML classification predictions with structured botanical knowledge base.
"""
import json
import logging
from typing import Dict, Any, List
from config import KNOWLEDGE_BASE_PATH

logger = logging.getLogger(__name__)

class RecommendationEngine:
    def __init__(self, kb_path=KNOWLEDGE_BASE_PATH):
        self.kb_path = kb_path
        self.knowledge_base = self._load_knowledge_base()

    def _load_knowledge_base(self) -> Dict[str, Any]:
        try:
            with open(self.kb_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading knowledge base from {self.kb_path}: {e}")
            return {}

    def get_recommendation(self, class_name: str, confidence: float, top_predictions: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generates full structured recommendation object based on predicted class and confidence.
        """
        profile = self.knowledge_base.get(class_name)
        if not profile:
            # Fallback to General Flower profile or generic advice
            profile = self.knowledge_base.get("General_Flower___Healthy", {
                "species": "Flowering Plant",
                "condition_name": "General Assessment",
                "category": "General",
                "severity": "mild",
                "description": "Standard flowering plant care profile.",
                "visual_symptoms": ["General leaf foliage."],
                "confidence_disclaimer": "Image diagnosis is an AI decision-support prediction.",
                "immediate_actions": ["Inspect foliage and adjust watering."],
                "watering": {"frequency": "As needed", "method": "Base watering", "schedule_tips": "Morning"},
                "sunlight_environment": {"sunlight": "Bright indirect light", "airflow": "Good", "temperature_humidity": "Moderate"},
                "soil_nutrition": {"soil_type": "Well draining", "ph_range": "6.0-7.0", "fertilizer": "Balanced"},
                "treatments": {"organic": ["Neem oil spray"], "chemical": ["None"]},
                "prevention": ["Sanitize tools", "Water at base"],
                "checklist_tasks": ["Check soil moisture", "Remove dead foliage"]
            })

        # Calibrate confidence text and alert level
        conf_percentage = round(confidence * 100, 1)
        if conf_percentage >= 80:
            reliability = "High Confidence"
            reliability_badge = "badge-success"
        elif conf_percentage >= 55:
            reliability = "Moderate Confidence"
            reliability_badge = "badge-warning"
        else:
            reliability = "Preliminary Suggestion"
            reliability_badge = "badge-info"

        # Severity badges
        severity = profile.get("severity", "mild")
        severity_map = {
            "healthy": {"label": "Healthy Foliage", "class": "severity-healthy", "icon": "fa-check-circle"},
            "mild": {"label": "Mild Attention Required", "class": "severity-mild", "icon": "fa-info-circle"},
            "moderate": {"label": "Action Needed", "class": "severity-moderate", "icon": "fa-exclamation-circle"},
            "severe": {"label": "Urgent Care Required", "class": "severity-severe", "icon": "fa-triangle-exclamation"}
        }
        severity_info = severity_map.get(severity, severity_map["mild"])

        # Format differential diagnosis note if secondary predictions are close
        differential_notes = []
        if top_predictions and len(top_predictions) > 1:
            for pred in top_predictions[1:3]:
                if pred["confidence"] > 0.15:
                    differential_notes.append(
                        f"Also consider **{pred['condition']}** ({round(pred['confidence']*100, 1)}% probability), as symptoms can visually overlap under certain watering or lighting conditions."
                    )

        return {
            "class_key": class_name,
            "species": profile.get("species", "Flowering Plant"),
            "condition_name": profile.get("condition_name", "Health Assessment"),
            "category": profile.get("category", "General"),
            "severity": severity,
            "severity_info": severity_info,
            "confidence_score": confidence,
            "confidence_percentage": conf_percentage,
            "reliability": reliability,
            "reliability_badge": reliability_badge,
            "description": profile.get("description", ""),
            "visual_symptoms": profile.get("visual_symptoms", []),
            "confidence_disclaimer": profile.get("confidence_disclaimer", ""),
            "differential_notes": differential_notes,
            "immediate_actions": profile.get("immediate_actions", []),
            "watering": profile.get("watering", {}),
            "sunlight_environment": profile.get("sunlight_environment", {}),
            "soil_nutrition": profile.get("soil_nutrition", {}),
            "treatments": profile.get("treatments", {}),
            "prevention": profile.get("prevention", []),
            "checklist_tasks": profile.get("checklist_tasks", []),
            "top_predictions": top_predictions or []
        }

    def ask_doctor(self, question: str, current_plant: str = "", current_condition: str = "") -> str:
        """
        Plant Doctor Q&A helper answering botanical and symptom inquiries.
        """
        q = question.lower()
        
        if "yellow" in q or "chlorosis" in q:
            return (
                "🌿 **Yellow Leaves Diagnosis Guide:**\n\n"
                "1. **Overwatering:** Soft, limp yellow lower leaves with wet soil → Let soil dry out 2-3 inches deep.\n"
                "2. **Iron Chlorosis:** Leaf blade is yellow but veins remain sharply dark green → Apply chelated iron spray and check soil pH (needs pH 6.0-6.5).\n"
                "3. **Nitrogen Deficiency:** Whole leaf uniformly pale yellow starting from oldest bottom leaves → Apply balanced organic fertilizer.\n"
                "4. **Underwatering:** Crispy dry yellow/brown leaf edges with bone-dry soil → Deep soak immediately."
            )
        elif "black spot" in q or "spot" in q:
            return (
                "🥀 **Leaf Spot Management:**\n\n"
                "• **Immediate Action:** Prune off all spotted leaves and discard in trash (never compost diseased foliage).\n"
                "• **Watering Rule:** Never wet leaves from above; water strictly at ground level early in the morning.\n"
                "• **Treatment:** Spray cold-pressed neem oil (1 oz/gallon) or liquid copper fungicide every 7-10 days."
            )
        elif "white" in q or "mildew" in q or "powder" in q:
            return (
                "🍂 **Powdery Mildew Treatment:**\n\n"
                "• **Homemade Remedy:** Mix 1 tbsp baking soda + 1/2 tsp liquid dish soap in 1 gallon warm water and spray weekly.\n"
                "• **Milk Spray:** 1 part milk to 9 parts water sprayed in morning sun creates natural antiseptic proteins.\n"
                "• **Environment:** Move plant to a spot with 6+ hours direct sun and strong airflow."
            )
        elif "bug" in q or "pest" in q or "aphid" in q or "insect" in q or "mite" in q:
            return (
                "🐛 **Pest Control Protocol:**\n\n"
                "• **Fast Knockdown:** Blast pests off stems with a sharp stream of water from garden hose.\n"
                "• **Organic Soap:** Spray insecticidal soap (2 tbsp pure castile soap per quart water) directly onto bug clusters.\n"
                "• **Neem Oil:** Apply 1-2% neem oil solution in late evening to stop pest reproductive cycles."
            )
        elif "water" in q or "how often" in q:
            return (
                "💧 **Watering Best Practices:**\n\n"
                "• Always use the **Finger Test**: Insert your index finger 1-2 inches into the soil. If dry, water deeply; if cool and damp, wait 1-2 days.\n"
                "• Always empty drainage trays 15 minutes after watering so roots don't sit in stagnant water.\n"
                "• Water in the morning so surface moisture evaporates before nighttime."
            )
        else:
            return (
                f"🌱 **Botanical Care Advice:**\n\n"
                f"For {current_plant or 'flowering plants'} experiencing {current_condition or 'health concerns'}, "
                "ensure 6+ hours of adequate light, verify soil drainage holes are unblocked, avoid wetting upper foliage, "
                "and apply organic preventative treatments like cold-pressed neem oil or compost tea."
            )
