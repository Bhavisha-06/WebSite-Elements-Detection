import wandb
from ultralytics import YOLO
from google.colab import drive

# Mount Google Drive to access the dataset
drive.mount("/content/drive")

# Initialize Weights & Biases (wandb) - set mode to "disabled" for now
wandb.init(mode="disabled")

# Load YOLOv10n model with the configuration file
yolov10n_model = YOLO("yolov10n.yaml")

# Train the YOLOv10n model on your dataset
yolov10n_model.train(
    data="/content/drive/My Drive/Webpage elements detection/data.yaml",  # Path to data configuration file
    epochs=50  # Number of training epochs
)

print("Training complete.")
