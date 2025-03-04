from transformers import AutoImageProcessor, HieraModel
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# Choose a valid model name for Hiera (for example, "facebook/hiera-base-224-hf").
model_name = "facebook/hiera-base-224-hf"  

# Load Hiera model and image processor (use fast processor and disable rescaling)
image_processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
model = HieraModel.from_pretrained(model_name)
model.eval()

# Define image transformations (ensure the ImageNet-1k preprocessing)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Hiera typically uses 224x224 input size
    transforms.ToTensor(),  # Convert image to Tensor with values in [0, 1]
])

# Load ImageNet-1K validation dataset
val_data_path = '/scratch/datasets/imagenet1k/val'  # Ensure this path points to ImageNet-1K validation data
val_dataset = datasets.ImageFolder(root=val_data_path, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

# Set up the device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Evaluate the model
correct = 0
total = 0

with torch.no_grad():
    for images, labels in tqdm(val_loader, desc="Evaluating", unit="batch"):
        images = images.to(device)  # Send images to the device (GPU/CPU)
        labels = labels.to(device)  # Send labels to the device

        # Preprocess the image using the AutoImageProcessor (set do_rescale=False to avoid rescaling again)
        inputs = image_processor(images, return_tensors="pt", do_rescale=False).to(device)

        # Forward pass through the model
        outputs = model(**inputs)

        # Extract the logits directly from the model output (assuming 'logits' exists)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs.last_hidden_state.mean(dim=1)

        # Apply softmax to logits if necessary
        probabilities = torch.nn.functional.softmax(logits, dim=1)

        # Get the predicted class
        _, predicted = torch.max(probabilities, 1)

        # Calculate accuracy
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = (correct / total) * 100
print(f"Accuracy on ImageNet-1k validation set: {accuracy:.2f}%")
