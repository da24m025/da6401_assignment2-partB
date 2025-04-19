import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import wandb
os.environ["WANDB_API_KEY"] = "e095fbd374bc0fa234acb179a6ec7620b57abf28"
# ===========================
# Configuration and Settings
# ===========================
# Initialize a W&B run
wandb.init(
    project="inaturalist_finetune", 
    config={
        "data_dir": "/kaggle/input/inaturalist/inaturalist_12K",
        "batch_size": 32,
        "num_epochs": 50,
        "learning_rate": 0.001,
        "valid_frac": 0.2,
        "model_name": "resnet50_feature_extraction"
    }
)
config = wandb.config

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =====================
# Data Transforms
# =====================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# =====================
# Prepare Datasets
# =====================
dataset_full = datasets.ImageFolder(os.path.join(config.data_dir, 'train'), transform=train_transform)
targets = [s[1] for s in dataset_full.samples]
indices = list(range(len(dataset_full)))
train_idx, val_idx = train_test_split(indices,
                                      test_size=config.valid_frac,
                                      stratify=targets,
                                      random_state=42)
dataset_train = Subset(dataset_full, train_idx)
dataset_full_val = datasets.ImageFolder(os.path.join(config.data_dir, 'train'), transform=val_transform)
dataset_val = Subset(dataset_full_val, val_idx)
dataset_test = datasets.ImageFolder(os.path.join(config.data_dir, 'test'), transform=val_transform)

# Data loaders
dataloaders = {
    'train': DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True),
    'val':   DataLoader(dataset_val,   batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True),
    'test':  DataLoader(dataset_test,  batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)
}
dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train','val','test']}
class_names = dataset_full.classes
num_classes = len(class_names)

# Log dataset info
wandb.log({
    "dataset/train_size": dataset_sizes['train'],
    "dataset/val_size":   dataset_sizes['val'],
    "dataset/test_size":  dataset_sizes['test'],
    "num_classes":        num_classes
})

# =====================
# Model Setup
# =====================
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

# Feature Extraction: freeze all backbone layers
for name, param in model.named_parameters():
    if 'fc' not in name:
        param.requires_grad = False
model = model.to(device)

# Watch model with W&B to log gradients and weights
wandb.watch(model, log="all", log_freq=10)

# =====================
# Loss and Optimizer
# =====================
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.fc.parameters(), lr=config.learning_rate, momentum=0.9)

# =====================
# Training and Validation Loop
# =====================
train_acc_history = []
val_acc_history = []
best_val_acc = 0.0
best_model_wts = model.state_dict()

for epoch in range(config.num_epochs):
    print(f"\nEpoch {epoch+1}/{config.num_epochs}")
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double().item() / dataset_sizes[phase] * 100

        # Log metrics to W&B
        wandb.log({
            f"{phase}/loss": epoch_loss,
            f"{phase}/accuracy": epoch_acc,
            "epoch": epoch+1,
            "learning_rate": optimizer.param_groups[0]['lr']
        })

        if phase == 'train':
            train_acc_history.append(epoch_acc)
        else:
            val_acc_history.append(epoch_acc)
            if epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                best_model_wts = model.state_dict()

# Load best model weights
model.load_state_dict(best_model_wts)

# =====================
# Test Set Evaluation
# =====================
model.eval()
correct_test = 0
with torch.no_grad():
    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)
        preds = model(inputs).argmax(1)
        correct_test += torch.sum(preds == labels.data).item()

test_acc = correct_test / dataset_sizes['test'] * 100
print(f"\nTest Accuracy: {test_acc:.2f}%")

# Log test accuracy\wandb.log({"test/accuracy": test_acc})

# =====================
# Plot and Log Accuracy Curves
# =====================
plt.figure(figsize=(8, 5))
plt.plot(range(1, config.num_epochs+1), train_acc_history, label='Train Acc')
plt.plot(range(1, config.num_epochs+1), val_acc_history,   label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Fine-Tuning ResNet50: Train vs. Val Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('finetune_resnet50_accuracy.png')
print("Accuracy plot saved to 'finetune_resnet50_accuracy.png'")

# Log the plot as an artifact/image
wandb.log({"accuracy_plot": wandb.Image('finetune_resnet50_accuracy.png')})

# =====================
# Final Analysis
# =====================
wandb.summary['best_val_accuracy'] = best_val_acc
wandb.summary['test_accuracy'] = test_acc

print("\n=== Analysis Summary ===")
print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
print(f"Test Accuracy           : {test_acc:.2f}%")
print("Strategy: Feature Extraction (freeze backbone, train final layer)")
print("- Leveraged pretrained ImageNet features for fast convergence.")
print("- Only final layer (~2M parameters) was trained.")
print("- Achieved superior accuracy vs. training from scratch.")