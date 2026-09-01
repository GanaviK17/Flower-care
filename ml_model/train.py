import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
from config import CLASS_NAMES, NUM_CLASSES, MODEL_PATH, MODEL_WEIGHTS_DIR
from ml_model.model import PlantHealthClassifier
from ml_model.preprocessor import get_training_transforms, get_inference_transforms
from ml_model.dataset_generator import DATASET_DIR, build_dataset, build_sample_gallery

class PlantDataset(Dataset):
    def __init__(self, root_dir: Path, class_names, transform=None):
        self.root_dir = root_dir
        self.class_names = class_names
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.transform = transform
        self.samples = []

        for cls_name in class_names:
            cls_dir = root_dir / cls_name
            if cls_dir.exists():
                for img_path in cls_dir.glob("*.jpg"):
                    self.samples.append((img_path, self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, target

def train_model(epochs: int = 5, batch_size: int = 16, lr: float = 0.002):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting model training on device: {device}", flush=True)

    if not DATASET_DIR.exists() or len(list(DATASET_DIR.glob("*/*.jpg"))) == 0:
        build_dataset(samples_per_class=10)
        build_sample_gallery()

    train_dataset = PlantDataset(DATASET_DIR, CLASS_NAMES, transform=get_training_transforms())
    val_dataset = PlantDataset(DATASET_DIR, CLASS_NAMES, transform=get_inference_transforms())

    print(f"Loaded {len(train_dataset)} training samples.", flush=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = PlantHealthClassifier(num_classes=NUM_CLASSES, pretrained=True, freeze_backbone=True)
    model.to(device)

    # Train only classifier parameters
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / total if total > 0 else 0
        epoch_acc = (correct / total) * 100 if total > 0 else 0

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                val_outputs = model(val_images)
                _, val_pred = val_outputs.max(1)
                val_total += val_labels.size(0)
                val_correct += val_pred.eq(val_labels).sum().item()

        val_acc = (val_correct / val_total) * 100 if val_total > 0 else 0
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.1f}% | Val Acc: {val_acc:.1f}%", flush=True)

    MODEL_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Trained model saved successfully to: {MODEL_PATH}", flush=True)

if __name__ == "__main__":
    train_model(epochs=5, batch_size=16)
