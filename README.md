![image](https://github.com/user-attachments/assets/43cbc6d3-43da-4227-846e-4cae67ad5f9c)
# Food Image Classifier with Nutritional Info

This project is a deep learning-based food image classifier that predicts the type of food in an image and displays its nutritional information. It uses a VGG16 model with transfer learning and supports both a Tkinter GUI and a web-based Flask interface for user-friendly testing.

---

## Demo

![Food Classifier Demo](food_Classifier_Demo.png)

---

## Features

- Classifies images into 11 food categories
- Displays nutrition details: Calories, Carbs, Fat, Protein
- Built-in GUI (Tkinter) and Web App (Flask)
- Visualizes confusion matrix
- Fine-tuning supported using transfer learning (VGG16)
- Easily extendable and interpretable

---

## Tech Stack

- **Python**
- **TensorFlow / Keras**
- **Scikit-learn**
- **Pillow (PIL)**
- **OpenCV**
- **Matplotlib & Seaborn**
- **Tkinter (for GUI)**
- **Flask (for web app)**

---

## Dataset Structure

```
food11/
├── train/
│   ├── apple_pie/
│   ├── cheesecake/
│   └── ...
└── test/
    ├── apple_pie/
    ├── cheesecake/
    └── ...
```

Supported classes:
`apple_pie`, `cheesecake`, `chicken_curry`, `french_fries`, `fried_rice`, `hamburger`, `hot_dog`, `ice_cream`, `omelette`, `pizza`, `sushi`

---

## Installation

1. Clone this repository:

```bash
git clone https://github.com/InsiyaFakhruddin/Food_Classifier.git
cd Food_Classifier
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install tensorflow keras numpy scikit-learn opencv-python pillow matplotlib seaborn flask
```

---

## How to Run

### Web App (Flask)

```bash
python app.py
```

- Navigate to `http://127.0.0.1:5000/`
- Upload a food image
- View the prediction and nutritional info

### Desktop GUI (Tkinter)

```bash
python main.py
```

- Choose and classify an image via a GUI window.

---

## Training the Model

The model is trained using transfer learning on `VGG16`. You can retrain or fine-tune it by calling:

```python
train_model(x_train, y_train, x_test, y_test)
fine_tune_model(model, datagen, x_train, y_train, x_test, y_test)
```

Model will be saved as:
- `food_classifier_1.keras` (initial)
- `fine_tuned_food_classifier.keras` (after tuning)

---

## Project Structure

```
├── app.py                      # Flask web interface
├── main.py                     # Tkinter GUI & training/testing logic
├── model/                      # Pretrained Keras model(s)
├── food11/                     # Dataset
├── Food Categories/            # Sample images for display
├── templates/ & static/        # Web frontend files
├── training_history.pkl        # Saved training metrics
├── utilities.py                # Helper methods
├── requirements.txt            # Dependency list
```

---

## Author
**Insiya Fakhruddin**  
AI & Deep Learning Enthusiast  
[GitHub](https://github.com/InsiyaFakhruddin)

---

## License

This project is licensed under the **MIT License** – feel free to use, modify, and share.

---

## Acknowledgements

- VGG16 from [Keras Applications](https://keras.io/api/applications/)
- Dataset: Food-11 (or your custom collection)
  
