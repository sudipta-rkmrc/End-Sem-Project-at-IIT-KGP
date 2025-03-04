"""
A trial code by Sudipta Sarkar.
Use Hiera on ImageNet-1k for evaluation.                    
Date: 25/02/2025                                         
Reference: https://dl.fbaipublicfiles.com/hiera/hiera_base_224.pth
           https://github.com/facebookresearch/hiera
"""
import torch
import torch.nn.functional as F  # For softmax activation
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from torchinfo import summary
from hiera import Hiera
# Load the model using torch.hub
#model = torch.hub.load("facebookresearch/hiera", model="hiera_base_224", pretrained=True, checkpoint="mae_in1k_ft_in1k")

model = Hiera.from_pretrained("facebook/hiera_base_224.mae_in1k_ft_in1k")  # mae pt then in1k ft'd model
model.save_pretrained("hiera-base-224", config=model.config)
# Set the model to evaluation mode
model.eval()

# Print the model summary
print(model)
summary(model, input_size=(512, 3, 224, 224))  # Show model summary for a batch of 512 images with 224x224 size

# Define image transformations (ensure the ImageNet-1k preprocessing)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224 (standard for Hiera models)
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

        # Forward pass through the model
        outputs = model(images)

        # Since the model directly returns logits, we can use them for classification
        logits = outputs  # The output is the logits

        # Apply softmax to get probabilities
        probabilities = F.softmax(logits, dim=1)

        # Get the predicted class by selecting the class with the highest probability
        _, predicted = torch.max(probabilities, 1)

        # Calculate accuracy
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = (correct / total) * 100
print(f"Accuracy on ImageNet-1k validation set: {accuracy:.2f}%")
