import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from pathlib import Path
from config import CLASS_NAMES, SAMPLE_IMAGES_FOLDER, BASE_DIR

DATASET_DIR = BASE_DIR / "ml_model" / "dataset"

def create_synthetic_plant_image(class_name: str, size=(256, 256), seed: int = None) -> Image.Image:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    parts = class_name.split("___")
    plant = parts[0]
    condition = parts[1] if len(parts) > 1 else "Healthy"

    # Base background
    bg_color = (random.randint(235, 245), random.randint(238, 248), random.randint(230, 240))
    img = Image.new("RGB", size, bg_color)
    draw = ImageDraw.Draw(img)

    w, h = size
    cx, cy = w // 2, h // 2

    # Plant-specific foliage colors
    if "Rose" in plant:
        leaf_color = (random.randint(25, 45), random.randint(95, 130), random.randint(30, 50))
        vein_color = (random.randint(40, 60), random.randint(130, 160), random.randint(40, 65))
        flower_color = (220, 20, 60)
    elif "Hibiscus" in plant:
        leaf_color = (random.randint(20, 40), random.randint(105, 145), random.randint(35, 55))
        vein_color = (random.randint(35, 55), random.randint(145, 180), random.randint(45, 70))
        flower_color = (255, 69, 0)
    elif "Marigold" in plant:
        leaf_color = (random.randint(30, 50), random.randint(90, 125), random.randint(25, 45))
        vein_color = (random.randint(45, 65), random.randint(120, 150), random.randint(35, 55))
        flower_color = (255, 165, 0)
    elif "Jasmine" in plant:
        leaf_color = (random.randint(15, 35), random.randint(85, 120), random.randint(25, 45))
        vein_color = (random.randint(30, 50), random.randint(120, 150), random.randint(35, 55))
        flower_color = (250, 250, 250)
    else:
        leaf_color = (random.randint(25, 45), random.randint(100, 135), random.randint(30, 50))
        vein_color = (random.randint(40, 60), random.randint(135, 165), random.randint(45, 65))
        flower_color = (218, 112, 214)

    # Disease foliage adjustments
    if "Chlorosis" in condition or "Nutrient" in condition:
        leaf_color = (random.randint(190, 220), random.randint(195, 225), random.randint(50, 80))
        vein_color = (random.randint(30, 55), random.randint(120, 150), random.randint(35, 55))
    elif "Overwatering" in condition:
        leaf_color = (random.randint(140, 170), random.randint(155, 185), random.randint(60, 90))
    elif "Dehydration" in condition:
        leaf_color = (random.randint(90, 115), random.randint(110, 135), random.randint(70, 90))

    # Main leaf
    leaf_bbox = [cx - 75, cy - 95, cx + 75, cy + 95]
    draw.ellipse(leaf_bbox, fill=leaf_color, outline=vein_color, width=2)

    # Veins
    draw.line([(cx, cy - 90), (cx, cy + 90)], fill=vein_color, width=3)
    for offset in range(-60, 70, 25):
        draw.line([(cx, cy + offset), (cx - 45, cy + offset - 15)], fill=vein_color, width=2)
        draw.line([(cx, cy + offset), (cx + 45, cy + offset - 15)], fill=vein_color, width=2)

    # Secondary leaves
    draw.ellipse([cx - 100, cy - 40, cx - 10, cy + 80], fill=leaf_color, outline=vein_color, width=1)
    draw.ellipse([cx + 10, cy - 40, cx + 100, cy + 80], fill=leaf_color, outline=vein_color, width=1)

    # Pathological features
    if "Black_Spot" in condition or "Leaf_Spot" in condition or "Alternaria" in condition or "Blight" in condition:
        for _ in range(random.randint(8, 16)):
            sx = cx + random.randint(-55, 55)
            sy = cy + random.randint(-70, 70)
            radius = random.randint(6, 14)
            draw.ellipse([sx - radius - 3, sy - radius - 3, sx + radius + 3, sy + radius + 3], fill=(210, 190, 40))
            draw.ellipse([sx - radius, sy - radius, sx + radius, sy + radius], fill=(30, 20, 15))

    elif "Powdery_Mildew" in condition:
        for _ in range(random.randint(12, 22)):
            mx = cx + random.randint(-60, 60)
            my = cy + random.randint(-75, 75)
            m_rad = random.randint(10, 25)
            draw.ellipse([mx - m_rad, my - m_rad, mx + m_rad, my + m_rad], fill=(240, 245, 245))

    elif "Rust" in condition:
        for _ in range(random.randint(20, 35)):
            rx = cx + random.randint(-60, 60)
            ry = cy + random.randint(-75, 75)
            r_rad = random.randint(3, 7)
            draw.ellipse([rx - r_rad - 2, ry - r_rad - 2, rx + r_rad + 2, ry + r_rad + 2], fill=(230, 200, 50))
            draw.ellipse([rx - r_rad, ry - r_rad, rx + r_rad, ry + r_rad], fill=(210, 95, 15))

    elif "Aphid" in condition:
        for _ in range(random.randint(25, 45)):
            ax = cx + random.randint(-25, 25)
            ay = cy - 80 + random.randint(-15, 35)
            a_rad = random.randint(2, 4)
            draw.ellipse([ax - a_rad, ay - a_rad, ax + a_rad, ay + a_rad], fill=(160, 210, 70))

    elif "Spider_Mites" in condition:
        for _ in range(60):
            px = cx + random.randint(-65, 65)
            py = cy + random.randint(-80, 80)
            draw.point((px, py), fill=(240, 230, 110))
        for _ in range(6):
            wx1 = cx + random.randint(-40, 40)
            wy1 = cy + random.randint(-60, 60)
            draw.line([(wx1, wy1), (wx1 + random.randint(-30, 30), wy1 + random.randint(-30, 30))], fill=(250, 250, 250), width=1)

    elif "Bud_Borer" in condition:
        draw.ellipse([cx - 20, cy - 100, cx + 20, cy - 60], fill=(180, 140, 100))
        draw.ellipse([cx - 5, cy - 85, cx + 5, cy - 75], fill=(30, 20, 10))

    elif "Dehydration" in condition:
        draw.chord([cx - 75, cy + 50, cx + 75, cy + 95], start=0, end=180, fill=(140, 90, 40))
        draw.chord([cx - 75, cy - 95, cx - 40, cy - 40], start=90, end=270, fill=(140, 90, 40))

    if "Healthy" in condition or random.random() > 0.4:
        fx, fy = cx + 60, cy - 60
        f_rad = 22
        draw.ellipse([fx - f_rad, fy - f_rad, fx + f_rad, fy + f_rad], fill=flower_color)
        draw.ellipse([fx - 8, fy - 8, fx + 8, fy + 8], fill=(255, 215, 0))

    img = img.filter(ImageFilter.SMOOTH_MORE)
    return img

def build_dataset(samples_per_class: int = 15):
    print(f"Generating dataset ({samples_per_class} per class)...")
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    for cls in CLASS_NAMES:
        cls_dir = DATASET_DIR / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(samples_per_class):
            img = create_synthetic_plant_image(cls, seed=i*100 + hash(cls) % 1000)
            img.save(cls_dir / f"img_{i:03d}.jpg", "JPEG", quality=92)
    print(f"Dataset generated at: {DATASET_DIR}")

def build_sample_gallery():
    print("Generating curated sample images for UI gallery...")
    SAMPLE_IMAGES_FOLDER.mkdir(parents=True, exist_ok=True)
    samples = [
        ("rose_healthy.jpg", "Rose___Healthy"),
        ("rose_black_spot.jpg", "Rose___Black_Spot"),
        ("rose_powdery_mildew.jpg", "Rose___Powdery_Mildew"),
        ("rose_aphids.jpg", "Rose___Aphid_Infestation"),
        ("hibiscus_healthy.jpg", "Hibiscus___Healthy"),
        ("hibiscus_blight.jpg", "Hibiscus___Leaf_Blight"),
        ("hibiscus_chlorosis.jpg", "Hibiscus___Iron_Chlorosis"),
        ("marigold_healthy.jpg", "Marigold___Healthy"),
        ("marigold_blight.jpg", "Marigold___Alternaria_Blight"),
        ("marigold_spider_mites.jpg", "Marigold___Spider_Mites"),
        ("jasmine_healthy.jpg", "Jasmine___Healthy"),
        ("jasmine_rust.jpg", "Jasmine___Leaf_Rust"),
        ("flower_overwatering.jpg", "General_Flower___Overwatering_Stress"),
        ("flower_dehydration.jpg", "General_Flower___Dehydration_Stress")
    ]
    for filename, cls in samples:
        img = create_synthetic_plant_image(cls, size=(300, 300), seed=42 + hash(cls) % 100)
        img.save(SAMPLE_IMAGES_FOLDER / filename, "JPEG", quality=95)
        print(f"Saved sample image: {filename}")

if __name__ == "__main__":
    build_dataset(15)
    build_sample_gallery()
