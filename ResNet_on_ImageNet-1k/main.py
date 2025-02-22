"""
A trial code by Sudipta Sarkar.
Use ResNet on ImageNet-1k.
Date: 21/02/2025
Reference: https://huggingface.co/docs/transformers/en/model_doc/resnet
"""
import torch
from transformers import AutoImageProcessor, ResNetForImageClassification
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# Load the pre-trained ResNet-50 model and image processor from Hugging Face
model_name = "microsoft/resnet-50"
image_processor = AutoImageProcessor.from_pretrained(model_name)
model = ResNetForImageClassification.from_pretrained(model_name)

# Define image preprocessing (transformations) to match the model's input requirements
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std),
])

# Load your custom validation dataset
val_data_path = '/scratch/datasets/imagenet1k/val'  # Path to your ImageNet-21k validation set
val_dataset = datasets.ImageFolder(root=val_data_path, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

# Set up the device for evaluation (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Evaluate the model on the validation set
correct = 0
total = 0

with torch.no_grad():
    for images, labels in tqdm(val_loader, desc="Evaluating", unit="batch"):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images).logits
        _, predicted = torch.max(outputs, 1)

        # Calculate accuracy
        total += labels.size(0)
        correct += (predicted == labels).sum().item()


accuracy = (correct / total) * 100
print(f"Accuracy on ImageNet-1k validation set: {accuracy:.2f}%")
