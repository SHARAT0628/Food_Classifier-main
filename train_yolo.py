import os
from ultralytics import YOLO

def train_food_model():
    # 1. Load a pre-trained YOLOv11 small model (suitable for RTX 3050)
    # 'yolo11s.pt' is the small version, 'yolo11n.pt' is nano.
    # We use 's' for better accuracy if the 3050 has 4GB+ VRAM.
    model = YOLO('yolo11s.pt') 

    print("--- Starting NutriVision v2 Training ---")
    print("Target: 60 Food Categories (Indian + Global)")
    
    # 2. Train the model
    # data: path to our dataset_config.yaml
    # epochs: 100 (adjust as needed)
    # imgsz: 640 (standard YOLO resolution)
    # batch: 16 (safe for 4GB VRAM, can increase to 32 if 6GB)
    results = model.train(
        data='dataset_config.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        device=0, # Use first GPU
        project='nutrivision_v2',
        name='food_detector',
        save=True
    )

    print("\n--- Training Complete ---")
    print(f"Best model saved at: {results.save_dir}/weights/best.pt")
    print("Please move 'best.pt' back to the main 'model/' folder of the project.")

if __name__ == '__main__':
    # Ensure dataset structure exists or alert the user
    if not os.path.exists('dataset_config.yaml'):
        print("Error: dataset_config.yaml not found!")
    else:
        train_food_model()
