import torch
import wandb

# Assuming model, device, and dataloaders are already defined
model.eval()
correct_test = 0
total_test_loss = 0.0
criterion = torch.nn.CrossEntropyLoss()  

with torch.no_grad():
    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(inputs)
        preds = outputs.argmax(1)
        
        # Compute accuracy
        correct_test += torch.sum(preds == labels.data).item()
        
        # Compute loss
        loss = criterion(outputs, labels)
        total_test_loss += loss.item() * inputs.size(0)

# Calculate metrics
test_acc = correct_test / dataset_sizes['test'] * 100
test_loss = total_test_loss / dataset_sizes['test']

# Print results
print(f"\nTest Accuracy: {test_acc:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Log metrics to W&B
wandb.log({
    "test/accuracy": test_acc,
    "test/loss": test_loss
})