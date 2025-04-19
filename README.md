# da6401_assignment2-partB
## DA24M025 -Teja Yelagandula

Fine-Tuning Pre-Trained ResNet50 on iNaturalist 12K Dataset
This project fine-tunes a pre-trained ResNet50 model on the iNaturalist 12K dataset for image classification. The model is fine-tuned using feature extraction, where only the final fully connected layer is trained, while the backbone layers are frozen. The project includes code for training, validation, and testing, with metrics logged to Weights & Biases (W&B) for tracking.
Project Structure

finetune_resnet50.py: The main script for fine-tuning the ResNet50 model on the iNaturalist 12K dataset.
log_test_metrics.py: A script to evaluate the model on the test set and log the test accuracy and loss to W&B.
finetune_resnet50_accuracy.png: A plot of training and validation accuracy curves generated during training.

Prerequisites

Kaggle Account: You need a Kaggle account to run the notebook.
Dataset: The iNaturalist 12K dataset must be uploaded as a zip folder to your Kaggle notebook.
W&B Account: A Weights & Biases account is required for logging metrics. Ensure you have your API key ready.

Instructions to Run
Since this project is designed to run on Kaggle, there are no specific commands involved. Follow these steps to execute the code consecutively in a Kaggle notebook:

Upload the Dataset:

Go to your Kaggle notebook.
Click on "+ Add Data" in the right sidebar.
Upload the iNaturalist 12K dataset as a zip folder.
Note the path where the dataset is mounted (typically /kaggle/input/inaturalist/inaturalist_12K).


Set Up the Notebook:

Create a new Kaggle notebook or upload an existing one.
Ensure the notebook has access to a GPU for faster training (Settings > Accelerator > GPU).


Install Required Libraries:

Run the following code in a notebook cell to install necessary libraries:!pip install wandb




Authenticate W&B:

Run the following code in a notebook cell and enter your W&B API key when prompted:import wandb
wandb.login()




Run the Training Script:

Copy and paste the code from finetune_resnet50.py into a notebook cell.
Adjust the data_dir path if necessary to match the dataset's location.
Execute the cell to start training. The script will:
Load and preprocess the dataset.
Fine-tune the ResNet50 model.
Log training and validation metrics to W&B.
Save the accuracy plot as finetune_resnet50_accuracy.png.




Run the Test Metrics Script:

After training, copy and paste the code from log_test_metrics.py into another notebook cell.
Ensure the model is loaded with the best weights (if saved).
Execute the cell to evaluate the model on the test set and log the test accuracy and loss to W&B.



Dataset
The iNaturalist 12K dataset should be organized as follows:
/kaggle/input/inaturalist/inaturalist_12K/
    train/
        class1/
            image1.jpg
            image2.jpg
        class2/
            image1.jpg
            image2.jpg
    test/
        class1/
            image1.jpg
        class2/
            image1.jpg


Train Directory: Contains subdirectories for each class with training images.
Test Directory: Contains subdirectories for each class with test images.

Model

Architecture: ResNet50 pre-trained on ImageNet.
Fine-Tuning Strategy: Feature extraction (freeze all backbone layers, train only the final fully connected layer).
Optimizer: SGD with a learning rate of 0.001 and momentum of 0.9.
Loss Function: Cross-Entropy Loss.

Results

Validation Accuracy: Approximately 77% after 50 epochs.
Test Accuracy: 78.10%.
Insights: Fine-tuning converges faster and achieves higher accuracy compared to training from scratch, leveraging pre-trained features effectively.

Logging with W&B

Project Name: inaturalist_finetune
Logged Metrics:
Training loss and accuracy.
Validation loss and accuracy.
Test loss and accuracy.
Accuracy plot.



Additional Notes

Early Stopping: Not implemented in this version. Consider adding it for better efficiency.
Hyperparameter Tuning: The current setup uses fixed hyperparameters. For further improvement, consider tuning learning rate, batch size, etc.
Model Saving: The best model weights are loaded for testing. Ensure to save them if needed for future use.

License
This project is licensed under the MIT License.
