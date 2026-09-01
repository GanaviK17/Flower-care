import io
from pathlib import Path
from typing import Tuple, Union
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import numpy as np

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_inference_transforms(image_size: Tuple[int, int] = (224, 224)):
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def get_training_transforms(image_size: Tuple[int, int] = (224, 224)):
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=25),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def load_and_preprocess_image(image_input: Union[str, Path, bytes, io.BytesIO, Image.Image], image_size: Tuple[int, int] = (224, 224)) -> Tuple[torch.Tensor, Image.Image]:
    if isinstance(image_input, (str, Path)):
        pil_img = Image.open(image_input)
    elif isinstance(image_input, bytes):
        pil_img = Image.open(io.BytesIO(image_input))
    elif isinstance(image_input, io.BytesIO):
        pil_img = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        pil_img = image_input
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    pil_img = ImageOps.exif_transpose(pil_img)
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    transform = get_inference_transforms(image_size)
    tensor = transform(pil_img).unsqueeze(0)
    return tensor, pil_img

def validate_image(pil_img: Image.Image) -> Tuple[bool, str]:
    w, h = pil_img.size
    if w < 64 or h < 64:
        return False, f"Image resolution too small ({w}x{h}). Minimum recommended is 128x128."
    img_np = np.array(pil_img)
    std = float(np.std(img_np))
    if std < 5.0:
        return False, "Uploaded image appears blank or lacks visual contrast."
    return True, "Valid"
