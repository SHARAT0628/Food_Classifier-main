# NutriVision: Large-Scale Multi-Regional Food Image Classification
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**NutriVision** is a robust deep learning framework engineered for automated cross-cultural dietary assessment. The system utilizes a heavily optimized **EfficientNetB0** architecture to classify over **136 distinct food categories** (combining Western staples and Indian cuisines) while synchronously providing macro-nutrient estimations (Calories, Proteins, Carbohydrates, Fats).

## 🚀 Key Features
* **Massive Dataset Support:** Trained on over 86,000 images uniformly pooled from the benchmark `Food-101` dataset and an explicit 35-variety Indian/Global collection.
* **Fault-Tolerant Dynamic Streaming:** Circumvents systemic memory (RAM) bottlenecks by utilizing a heavily customized parallel generation component that filters file corruption at runtime.
* **High-Efficiency Transfer Learning:** Extracts features via ImageNet weights and systematically fine-tunes custom projection kernels across 243 functional layers using a dual-phase training protocol.
* **Academic Glassmorphism Interface:** Integrates a clinical, responsive Flask frontend for real-time model invocation, nutritional mapping, and diagnostic readout formatting.

## 📁 Repository Structure
```text
├── app.py                       # Flask Inference Server & UI Engine
├── train_model_optimized.py     # Fault-Tolerant Training Mechanics
├── model/                       # [IGNORED BY GIT] Dynamic Weights (.keras)
├── templates/                   # Frontend UI components (index.html)
└── requirements.txt             # Dependency Definitions
```

## 🚀 Quick Start (GitHub Deployment)

To run this application on a new system utilizing our GitHub codebase, please closely follow these instructions. 
> **⚠️ CRITICAL:** To bypass harsh Git capacity limits, the Neural Network parameters (`model/food_classifier_combined.keras`) and the 6GB raw image datasets are **intentionally ignored** via `.gitignore`. 

### Step 1: Clone the Repository
Open a terminal on the deployment system and execute:
```bash
git clone https://github.com/SHARAT0628/Food_Classifier-main.git
cd Food_Classifier-main
```

### Step 2: Manually Inject the Model
Since GitHub strictly filters dense binary matrices, you must manually introduce the generated model weights:
1. Ensure you have copied the explicitly trained `model/` folder (containing the `.keras` model and JSON mappings) from your host computer via an external drive.
2. Paste the `model/` folder directly into the root directory of the freshly cloned repository (right next to `app.py`).

### Step 3: Install Dependencies
The application requires specific Python machine-learning parameters. Run:
```bash
pip install -r requirements.txt
```

### Step 4: Launch the Interface
Boot the diagnostic web inference server:
```bash
python app.py
```
Finally, access `http://127.0.0.1:5000` via any standard browser to upload images and utilize the diagnostic GUI!
