import os
from ultralytics import YOLO

def train():
    print("Starting YOLOv8 training pipeline...")
    
    model = YOLO("yolov8n.pt")  # Start from pretrained weights
    
    # Check if dataset mapping exists
    yaml_path = os.path.abspath("../dataset/dataset.yaml")
    if not os.path.exists(yaml_path):
        # Depending on cwd, try relative to current
        yaml_path = os.path.abspath("dataset/dataset.yaml")
        
    if not os.path.exists(yaml_path):
        print(f"Error: dataset.yaml not found at {yaml_path}")
        print("Dataset configuration is missing. Training cannot proceed.")
        return

    print(f"Training on dataset configuration: {yaml_path}")
    
    results = model.train(
        data=yaml_path,
        epochs=50,
        batch=16,
        imgsz=640,
        name='helmet_yolov8n',
        device='0' # Use gpu if available, otherwise switch to 'cpu'
    )
    
    print("Training finished.")
    print("The best weights have been saved under runs/detect/helmet_yolov8n/weights/best.pt")

if __name__ == "__main__":
    train()
