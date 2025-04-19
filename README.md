# Fine-Tuning ResNet50 on iNaturalist-12K (Part B)

**Author:** Teja Yelagandula  
**Institute ID:** DA24M025  
**GitHub Repository:** https://github.com/da24m025/da6401_assignment2-partB  
**W&B Report:** [ DA6401 Assignment 2 W&B Report](https://wandb.ai/fgbb66579-iit-madras-foundation/inaturalist_cnn_from_scratch3988/reports/Teja-Yelagandula-DA6401-Assignment-2--VmlldzoxMjE0MDAxNw?accessToken=a0qcojswservt8etlc43cdfwh0n5vrtchgehy3btabcth6eirhpcsgdl5l11133k)

---

## 📖 Project Overview

This project demonstrates a **feature-extraction** fine-tuning strategy on a pre-trained ResNet50 model using the iNaturalist-12K dataset. Only the final classification layer is trained, leveraging ImageNet weights for rapid convergence and strong accuracy on species classification.

---

## 📁 Repository Structure

```
├── finetune_resnet50.py       # Fine-tuning script (training & validation)
├── log_test_metrics.py        # Test-set evaluation and W&B logging
├── DL-Project Part B.ipynb     # Reference notebook with end-to-end workflow
├── finetune_resnet50_accuracy.png  # Generated accuracy plot artifact
└── README.md                  # Project documentation (this file)
```

### File Dependencies
- All scripts are self-contained but may import standard libraries (`torch`, `torchvision`, `wandb`, etc.).
- The dataset must be organized under a `train/` and `test/` directory for `ImageFolder`.

---

##  Running in Kaggle Environment

1. **Open a new Kaggle Notebook** and upload this repository’s files.  
2. **Mount the iNaturalist-12K dataset** such that:
   ```text
   /kaggle/working/data/train/...
   /kaggle/working/data/test/...
   ```
3. **Use `DL-Project Part B.ipynb`** to:
   - Preprocess and load data.  
   - Copy and run `finetune_resnet50.py` cells to fine-tune ResNet50.  
   - Copy and run `log_test_metrics.py` cells to evaluate on test set.  
   - Visualize training & validation curves and inspect metrics.

_All execution is performed via notebook cells; no CLI steps are required._

---

##  Code Description

### finetune_resnet50.py
- Loads `torchvision.models.resnet50(pretrained=True)`.
- Replaces `fc` layer with a new `nn.Linear` matching the number of classes.
- Freezes all backbone parameters (feature-extraction strategy).
- Implements training & validation loops with W&B logging (`wandb.log`).
- Saves best model weights and generates an accuracy plot (`wandb.Image`).

### log_test_metrics.py
- Loads the best checkpoint from `finetune_resnet50.py` training.
- Evaluates the model on the held-out `test/` set.
- Logs final test accuracy and loss to W&B (`wandb.log`).

---

## 📊 Dataset Structure

The iNaturalist-12K dataset should be arranged as:
```
/kaggle/working/data/
├── train/
│   ├── class1/ *.jpg
│   ├── class2/ *.jpg
│   └── ...
└── test/
    ├── class1/ *.jpg
    ├── class2/ *.jpg
    └── ...
```

---

## 🏗 Model & Hyperparameters

- **Base Model:** ResNet50 pretrained on ImageNet
- **Fine-Tuning Strategy:** Freeze all layers except final `fc`
- **Optimizer:** SGD (`lr=0.001, momentum=0.9`)
- **Loss:** CrossEntropyLoss
- **Batch Size:** 32
- **Epochs:** 50

---

## 📈 Results Summary

- **Validation Accuracy:** ~77% after 50 epochs
- **Test Accuracy:** 78.10%
- **Convergence:** Rapid improvement in first 10 epochs, stable thereafter.

---

##  Future Work

- Add **early stopping** to terminate when validation plateaus.
- Integrate **W&B Sweeps** for hyperparameter optimization.
- Experiment with **data augmentation** techniques (MixUp, CutMix).
- Explore **partial fine-tuning** by unfreezing deeper layers.





