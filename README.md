# da6401_assignment2-partB
## DA24M025 -Teja Yelagandula

# Fine-Tuning ResNet50 on iNaturalist-12K


##  Project Overview
This repository contains code and resources for fine-tuning a pre-trained ResNet50 model on the iNaturalist-12K dataset. We employ a feature-extraction strategy—freezing the backbone layers and training only the final classification layer—to achieve efficient convergence and strong performance on the species classification task.

##  Key Features
- **Feature Extraction**: Freeze all ResNet50 backbone parameters and fine-tune only the final fully connected layer.
- **Modular Scripts**:
  - `finetune_resnet50.py`: Training and validation loop with W&B logging.
  - `log_test_metrics.py`: Evaluation on the held-out test set and W&B logging of final metrics.
- **W&B Integration**: Track training/validation loss & accuracy, test metrics, and visualize performance curves.
- **Results Visualization**: Automatically save training/validation accuracy plots (`finetune_resnet50_accuracy.png`).

## 📁 Project Structure
```
├── finetune_resnet50.py          # Main training & validation script
├── log_test_metrics.py           # Script for test set evaluation and W&B logging
├── finetune_resnet50_accuracy.png # Generated training vs validation accuracy plot
├── README.md                     # This documentation file
```

##  Prerequisites
- **Kaggle Account**: To run notebooks on Kaggle.
- **W&B Account**: For experiment tracking. Retrieve your API key from https://wandb.ai/
- **GPU Runtime**: Recommended for faster training.

##  Getting Started
1. **Clone the repository** (if running locally):
   ```bash
   git clone https://github.com/your-username/inaturalist_finetune.git
   cd inaturalist_finetune
   ```
2. **Upload the Dataset on Kaggle**:
   - In your Kaggle notebook, click **+ Add Data** → **Upload**, and select the iNaturalist-12K ZIP archive.
   - Note the mount path (e.g., `/kaggle/input/inaturalist/inaturalist_12K`).
3. **Install Dependencies**:
   ```bash
   pip install torch torchvision wandb
   ```
4. **Authenticate W&B**:
   ```python
   import wandb
   wandb.login()
   ```
5. **Run Training**:
   - Paste the contents of `finetune_resnet50.py` into a Kaggle notebook cell.
   - Update `data_dir` to your dataset path.
   - Execute to start training. Metrics and model checkpoints will be logged to W&B, and an accuracy plot will be saved.

6. **Evaluate on Test Set**:
   - Paste `log_test_metrics.py` into a new cell.
   - Ensure the script loads the best model weights.
   - Execute to compute and log test accuracy & loss to W&B.

##  Dataset Organization
The iNaturalist-12K dataset should follow this structure:
```
/kaggle/input/inaturalist/inaturalist_12K/
├── train/
│   ├── class1/
│   │   ├── img001.jpg
│   │   └── img002.jpg
│   ├── class2/
│   │   └── img001.jpg
│   └── ...
└── test/
    ├── class1/
    │   └── img010.jpg
    └── class2/
        └── img005.jpg
```

##  Model Architecture & Hyperparameters
- **Base Model**: `torchvision.models.resnet50(pretrained=True)`
- **Frozen Layers**: All layers except the final `fc` layer
- **Optimizer**: SGD (`lr=0.001`, `momentum=0.9`)
- **Loss Function**: CrossEntropyLoss
- **Batch Size**: 32
- **Epochs**: 50

##  Results
- **Validation Accuracy**: ~77% after 50 epochs
- **Test Accuracy**: 78.10%
- **Convergence**: Feature extraction yields faster convergence than training from scratch


##  Logging & Visualization
- **W&B Project**: `inaturalist_finetune`
- **Tracked Metrics**:
  - Training & validation loss
  - Training & validation accuracy
  - Test loss & accuracy
- **Artifacts**: Accuracy plots and model checkpoints

##  Future Improvements
- **Early Stopping**: Automatically halt training when validation performance plateaus.
- **Hyperparameter Search**: Integrate W&B Sweeps for tuning learning rate, batch size, and optimizer types.
- **Data Augmentation**: Explore advanced transforms (CutMix, MixUp) to improve generalization.


