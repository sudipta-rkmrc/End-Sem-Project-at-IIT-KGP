"""
A trial code by Sudipta Sarkar.
Use Hiera on ImageNet-1k for evaluation.                    
Date: 25/02/2025                                         
Reference: https://github.com/facebookresearch/mvit
"""

"""
from transformers import AutoImageProcessor, HieraModel
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# Choose a valid model name for Hiera (for example, "facebook/hiera-tiny-224-hf").
model_name = "facebook/hiera-base-224-hf"  

# Load Hiera model and image processor (which includes preprocessing and augmentation)
image_processor = AutoImageProcessor.from_pretrained(model_name, do_rescale=False)
model = HieraModel.from_pretrained(model_name)
model.eval()


print(model)

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

        # Preprocess the image using the AutoImageProcessor
        inputs = image_processor(images, return_tensors="pt").to(device)

        # Forward pass through the model
        outputs = model(**inputs)

        # Inspect the outputs to identify where the logits are located
        #print(outputs)  # Uncomment this line to inspect the output structure

        # Check if 'logits' exists in the output and use that for classification
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            # If no 'logits', we might use another output (such as 'last_hidden_state')
            # You may need to use the hidden state or another output, depending on your model
            logits = outputs.last_hidden_state.mean(dim=1)  # Example: use mean of the hidden states

        # Get the predicted class
        _, predicted = torch.max(logits, 1)

        # Calculate accuracy
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = (correct / total) * 100
print(f"Accuracy on ImageNet-1k validation set: {accuracy:.2f}%")
"""







from transformers import AutoImageProcessor, HieraModel
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchinfo import summary
import torch.nn.functional as F  # For softmax activation

# Choose a valid model name for Hiera (for example, "facebook/hiera-base-224-hf").
model_name = "facebook/hiera-base-224-hf"  

# Load Hiera model and image processor (which includes preprocessing and augmentation)
image_processor = AutoImageProcessor.from_pretrained(model_name, do_rescale=False)
model = HieraModel.from_pretrained(model_name)
model.eval()

# Modify the output layer explicitly to 1000 classes (ImageNet-1K)
# We assume the model has a final classification head, typically it might be 'head' or 'classifier'
if hasattr(model, 'head'):  # If the model uses 'head' for the final classification layer
    model.head = torch.nn.Linear(model.head.in_features, 1000)
elif hasattr(model, 'classifier'):  # If it uses 'classifier' for the final classification layer
    model.classifier = torch.nn.Linear(model.classifier.in_features, 1000)
else:
    # If no specific layer found, it might be something custom
    # Manually add the final classification layer
    model.classifier = torch.nn.Linear(model.config.hidden_size, 1000)  # Set to 1000 classes

# Define image transformations (ensure the ImageNet-1k preprocessing)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Hiera typically uses 224x224 input size
    transforms.ToTensor(),  # Convert image to Tensor with values in [0, 1]
])

# Print the model summary
print(model)
summary(model, input_size=(512, 3, 224, 224))

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

        # Preprocess the image using the AutoImageProcessor
        inputs = image_processor(images, return_tensors="pt").to(device)

        # Forward pass through the model
        outputs = model(**inputs)

        # Check if 'logits' exists in the output and use that for classification
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            # If no 'logits', we might use another output (such as 'last_hidden_state')
            # You may need to use the hidden state or another output, depending on your model
            logits = outputs.last_hidden_state.mean(dim=1)  # Example: use mean of the hidden states

        # Apply softmax to get probabilities
        probabilities = F.softmax(logits, dim=1)

        # Get the predicted class by selecting the class with the highest probability
        _, predicted = torch.max(probabilities, 1)

        # Calculate accuracy
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = (correct / total) * 100
print(f"Accuracy on ImageNet-1k validation set: {accuracy:.2f}%")
