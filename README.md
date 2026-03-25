# NutriVision: Large-Scale Multi-Regional Food Image Classification
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**NutriVision** is a robust deep learning framework engineered for automated cross-cultural dietary assessment. The system utilizes a heavily optimized **EfficientNetB0** architecture to classify over **136 distinct food categories** (combining Western staples and Indian cuisines) while synchronously providing macro-nutrient estimations (Calories, Proteins, Carbohydrates, Fats).

## 🚀 Key Features
* **Massive Dataset Support:** Trained on over 86,000 images uniformly pooled from the benchmark `Food-101` dataset and an explicit 35-variety Indian/Global collection.
* **Fault-Tolerant Dynamic Streaming:** Circumvents systemic memory (RAM) bottlenecks by utilizing a heavily customized parallel generation component that filters file corruption at runtime.
* **High-Efficiency Transfer Learning:** Extracts features via ImageNet weights and systematically fine-tunes custom projection kernels across 243 functional layers using a dual-phase training protocol.
* **Academic Glassmorphism Interface:** Integrates a clinical, responsive Flask frontend for real-time model invocation, nutritional mapping, and diagnostic readout formatting.

## 🛠 Project Architecture
NutriVision abandons exhaustive static memory allocation for an active runtime pipeline.
1. **Inputs:** `(224, 224, 3)` Tensors dynamically loaded by a sequential disk streamer.
2. **Backbone:** Convolutional `EfficientNetB0` executing standard internal `Rescaling(1./255)` blocks against pure byte-level inputs.
3. **Classification Head:** 
    - `GlobalAveragePooling2D`
    - Dual `Dropout(0.3)` layers wrapping a high-capacity `Dense(512)` matrix.
    - Softmax `Dense(136)` multi-regional classifier payload.

## 📁 Repository Structure
```text
├── app.py                       # Flask Inference Server & UI Engine
├── train_model_optimized.py     # Fault-Tolerant Training Mechanics
├── model/                       # Dynamically Saved Weights & Mapping (.keras)
├── templates/                   # Frontend UI components (index.html)
└── research/                    # Extensive IEEE specifications and diagrams
```

## 🧠 Academic Context (IEEE Access)
This repository contains the codebase utilized for compiling empirical calculations, parameter depth, and dataset processing formulas designed for academic publication. You can locate explicitly structured diagrams, table formatting, and methodologies strictly within the `research/` directory.

## 🚀 Quick Start

### 1. Training the Convolutional Framework
Ensure your datasets are downloaded and extracted into the `food-101/` and `food-35/` folders. Then, launch the memory-efficient streaming pipeline:
```bash
python train_model_optimized.py
```

### 2. Launching the Diagnostics Server
Once the combined checkpoint (`model/food_classifier_combined.keras`) exists, launch the visual inference tool:
```bash
python app.py
```
Navigate to `http://127.0.0.1:5000` to utilize the diagnostic interface.
