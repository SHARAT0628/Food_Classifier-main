# NutriVision v2 - Training Guide

This guide is for training the **Object Detection** model for food recognition (60 classes, Indian + Global).

## Setup Instructions (Run on Laptop with NVIDIA GPU)

### 1. Install Dependencies
Open a terminal (PowerShell or CMD) in this folder and run:
```powershell
pip install -r requirements_training.txt
```
*Note: This will install torch with CUDA support for your RTX 3050.*

### 2. Prepare the Dataset
The `train_yolo.py` expects a folder named `datasets/food_data` with the following structure:
```text
datasets/food_data/
  images/
    train/ (image files)
    val/ (image files)
  labels/
    train/ (yolo .txt files)
    val/ (yolo .txt files)
```
You can download a YOLO-formatted dataset from **Roboflow** (search for "Food-102" or "Indian Food Detection") and extract it here.

### 3. Start Training
Run the training script:
```powershell
python train_yolo.py
```
*Tip: If you get a 'CUDA out of memory' error, open `train_yolo.py` and change `batch=16` to `batch=8`.*

### 4. Transfer the Model
Once training finishes (it will take a few hours), look for a folder named `nutrivision_v2/food_detector/weights/`. 
1. Copy the **`best.pt`** file.
2. Send it back to the main project's **`model/`** folder.

---
**Happy Training!**
