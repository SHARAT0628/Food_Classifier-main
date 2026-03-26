from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
from keras._tf_keras.keras.models import load_model
import os
import json

app = Flask(__name__)

# ── Model loading ────────────────────────────────────────────────────
# Try combined model first, fallback to old model
MODEL_PATH = 'model/food_classifier_combined.keras'
CATEGORIES_PATH = 'model/categories.json'

if os.path.exists(MODEL_PATH) and os.path.exists(CATEGORIES_PATH):
    model = load_model(MODEL_PATH)
    with open(CATEGORIES_PATH, 'r') as f:
        CATEGORIES = json.load(f)
    print(f"[✓] Loaded combined model with {len(CATEGORIES)} categories")
elif os.path.exists('model/fine_tuned_food_classifier.keras'):
    model = load_model('model/fine_tuned_food_classifier.keras')
    CATEGORIES = ['apple_pie', 'cheesecake', 'chicken_curry', 'french_fries', 'fried_rice',
                  'hamburger', 'hot_dog', 'ice_cream', 'omelette', 'pizza', 'sushi']
    print("[!] Using old fine-tuned model (11 categories)")
elif os.path.exists('model/food_classifier.keras'):
    model = load_model('model/food_classifier.keras')
    CATEGORIES = ['apple_pie', 'cheesecake', 'chicken_curry', 'french_fries', 'fried_rice',
                  'hamburger', 'hot_dog', 'ice_cream', 'omelette', 'pizza', 'sushi']
    print("[!] Using old base model (11 categories)")
elif os.path.exists('fine_tuned_food_classifier.keras'):
    model = load_model('fine_tuned_food_classifier.keras')
    CATEGORIES = ['apple_pie', 'cheesecake', 'chicken_curry', 'french_fries', 'fried_rice',
                  'hamburger', 'hot_dog', 'ice_cream', 'omelette', 'pizza', 'sushi']
    print("[!] Using fine-tuned model from root (11 categories)")
else:
    print("[✗] No model found! Train a model first with: python train_model.py")
    model = None
    CATEGORIES = []

TARGET_SIZE = (224, 224)

# ── Nutritional Info (per serving, approximate) ─────────────────────
# Covers Food-101 categories + 35 Indian/Western varieties
NUTRIENTS = {
    # === Food-101 Categories ===
    'apple_pie': {'calories': 237, 'protein': 2, 'carbs': 34, 'fat': 11},
    'baby_back_ribs': {'calories': 290, 'protein': 22, 'carbs': 5, 'fat': 21},
    'baklava': {'calories': 334, 'protein': 5, 'carbs': 38, 'fat': 19},
    'beef_carpaccio': {'calories': 180, 'protein': 18, 'carbs': 2, 'fat': 11},
    'beef_tartare': {'calories': 200, 'protein': 20, 'carbs': 2, 'fat': 12},
    'beet_salad': {'calories': 90, 'protein': 2, 'carbs': 14, 'fat': 3},
    'beignets': {'calories': 320, 'protein': 5, 'carbs': 42, 'fat': 15},
    'bibimbap': {'calories': 490, 'protein': 25, 'carbs': 62, 'fat': 15},
    'bread_pudding': {'calories': 290, 'protein': 6, 'carbs': 43, 'fat': 11},
    'breakfast_burrito': {'calories': 350, 'protein': 15, 'carbs': 35, 'fat': 16},
    'bruschetta': {'calories': 120, 'protein': 3, 'carbs': 16, 'fat': 5},
    'caesar_salad': {'calories': 200, 'protein': 8, 'carbs': 10, 'fat': 14},
    'cannoli': {'calories': 230, 'protein': 5, 'carbs': 25, 'fat': 12},
    'caprese_salad': {'calories': 180, 'protein': 10, 'carbs': 4, 'fat': 14},
    'carrot_cake': {'calories': 310, 'protein': 3, 'carbs': 44, 'fat': 14},
    'ceviche': {'calories': 130, 'protein': 15, 'carbs': 10, 'fat': 3},
    'cheese_plate': {'calories': 350, 'protein': 18, 'carbs': 5, 'fat': 28},
    'cheesecake': {'calories': 321, 'protein': 5, 'carbs': 26, 'fat': 22},
    'chicken_curry': {'calories': 300, 'protein': 22, 'carbs': 14, 'fat': 18},
    'chicken_quesadilla': {'calories': 400, 'protein': 22, 'carbs': 30, 'fat': 20},
    'chicken_wings': {'calories': 320, 'protein': 27, 'carbs': 8, 'fat': 20},
    'chocolate_cake': {'calories': 350, 'protein': 4, 'carbs': 50, 'fat': 16},
    'chocolate_mousse': {'calories': 260, 'protein': 4, 'carbs': 28, 'fat': 16},
    'churros': {'calories': 280, 'protein': 3, 'carbs': 38, 'fat': 14},
    'clam_chowder': {'calories': 200, 'protein': 8, 'carbs': 18, 'fat': 11},
    'club_sandwich': {'calories': 400, 'protein': 25, 'carbs': 30, 'fat': 20},
    'crab_cakes': {'calories': 220, 'protein': 14, 'carbs': 12, 'fat': 13},
    'creme_brulee': {'calories': 310, 'protein': 4, 'carbs': 32, 'fat': 18},
    'croque_madame': {'calories': 480, 'protein': 22, 'carbs': 28, 'fat': 30},
    'cup_cakes': {'calories': 270, 'protein': 3, 'carbs': 38, 'fat': 12},
    'deviled_eggs': {'calories': 130, 'protein': 6, 'carbs': 1, 'fat': 11},
    'donuts': {'calories': 280, 'protein': 4, 'carbs': 33, 'fat': 15},
    'dumplings': {'calories': 230, 'protein': 10, 'carbs': 28, 'fat': 8},
    'edamame': {'calories': 120, 'protein': 12, 'carbs': 9, 'fat': 5},
    'eggs_benedict': {'calories': 420, 'protein': 20, 'carbs': 26, 'fat': 26},
    'escargots': {'calories': 170, 'protein': 14, 'carbs': 2, 'fat': 12},
    'falafel': {'calories': 330, 'protein': 13, 'carbs': 32, 'fat': 18},
    'filet_mignon': {'calories': 280, 'protein': 30, 'carbs': 0, 'fat': 17},
    'fish_and_chips': {'calories': 500, 'protein': 22, 'carbs': 50, 'fat': 23},
    'foie_gras': {'calories': 460, 'protein': 7, 'carbs': 4, 'fat': 44},
    'french_fries': {'calories': 312, 'protein': 3, 'carbs': 41, 'fat': 15},
    'french_onion_soup': {'calories': 300, 'protein': 12, 'carbs': 22, 'fat': 18},
    'french_toast': {'calories': 280, 'protein': 8, 'carbs': 36, 'fat': 12},
    'fried_calamari': {'calories': 270, 'protein': 14, 'carbs': 22, 'fat': 14},
    'fried_rice': {'calories': 340, 'protein': 10, 'carbs': 48, 'fat': 12},
    'frozen_yogurt': {'calories': 160, 'protein': 4, 'carbs': 30, 'fat': 3},
    'garlic_bread': {'calories': 200, 'protein': 4, 'carbs': 24, 'fat': 10},
    'gnocchi': {'calories': 250, 'protein': 5, 'carbs': 40, 'fat': 8},
    'greek_salad': {'calories': 150, 'protein': 5, 'carbs': 10, 'fat': 11},
    'grilled_cheese_sandwich': {'calories': 370, 'protein': 14, 'carbs': 28, 'fat': 23},
    'grilled_salmon': {'calories': 250, 'protein': 30, 'carbs': 0, 'fat': 14},
    'guacamole': {'calories': 150, 'protein': 2, 'carbs': 8, 'fat': 13},
    'gyoza': {'calories': 200, 'protein': 8, 'carbs': 24, 'fat': 8},
    'hamburger': {'calories': 400, 'protein': 22, 'carbs': 30, 'fat': 22},
    'hot_and_sour_soup': {'calories': 120, 'protein': 6, 'carbs': 12, 'fat': 5},
    'hot_dog': {'calories': 290, 'protein': 11, 'carbs': 22, 'fat': 18},
    'huevos_rancheros': {'calories': 350, 'protein': 16, 'carbs': 28, 'fat': 20},
    'hummus': {'calories': 180, 'protein': 7, 'carbs': 15, 'fat': 10},
    'ice_cream': {'calories': 270, 'protein': 4, 'carbs': 32, 'fat': 14},
    'lasagna': {'calories': 380, 'protein': 20, 'carbs': 32, 'fat': 18},
    'lobster_bisque': {'calories': 280, 'protein': 12, 'carbs': 14, 'fat': 20},
    'lobster_roll_sandwich': {'calories': 350, 'protein': 18, 'carbs': 28, 'fat': 18},
    'macaroni_and_cheese': {'calories': 400, 'protein': 15, 'carbs': 40, 'fat': 20},
    'macarons': {'calories': 150, 'protein': 2, 'carbs': 20, 'fat': 7},
    'miso_soup': {'calories': 60, 'protein': 4, 'carbs': 6, 'fat': 2},
    'mussels': {'calories': 170, 'protein': 24, 'carbs': 7, 'fat': 4},
    'nachos': {'calories': 350, 'protein': 10, 'carbs': 35, 'fat': 18},
    'omelette': {'calories': 250, 'protein': 16, 'carbs': 2, 'fat': 20},
    'onion_rings': {'calories': 330, 'protein': 5, 'carbs': 40, 'fat': 17},
    'oysters': {'calories': 80, 'protein': 8, 'carbs': 5, 'fat': 3},
    'pad_thai': {'calories': 400, 'protein': 15, 'carbs': 50, 'fat': 15},
    'paella': {'calories': 350, 'protein': 18, 'carbs': 42, 'fat': 12},
    'pancakes': {'calories': 250, 'protein': 6, 'carbs': 38, 'fat': 8},
    'panna_cotta': {'calories': 280, 'protein': 4, 'carbs': 28, 'fat': 18},
    'peking_duck': {'calories': 320, 'protein': 22, 'carbs': 10, 'fat': 22},
    'pho': {'calories': 350, 'protein': 20, 'carbs': 40, 'fat': 10},
    'pizza': {'calories': 300, 'protein': 12, 'carbs': 34, 'fat': 12},
    'pork_chop': {'calories': 280, 'protein': 30, 'carbs': 0, 'fat': 17},
    'poutine': {'calories': 450, 'protein': 15, 'carbs': 45, 'fat': 24},
    'prime_rib': {'calories': 350, 'protein': 28, 'carbs': 0, 'fat': 26},
    'pulled_pork_sandwich': {'calories': 400, 'protein': 22, 'carbs': 35, 'fat': 18},
    'ramen': {'calories': 450, 'protein': 18, 'carbs': 55, 'fat': 16},
    'ravioli': {'calories': 300, 'protein': 12, 'carbs': 35, 'fat': 12},
    'red_velvet_cake': {'calories': 340, 'protein': 4, 'carbs': 46, 'fat': 16},
    'risotto': {'calories': 320, 'protein': 8, 'carbs': 45, 'fat': 12},
    'samosa': {'calories': 260, 'protein': 5, 'carbs': 28, 'fat': 14},
    'sashimi': {'calories': 130, 'protein': 22, 'carbs': 0, 'fat': 4},
    'scallops': {'calories': 150, 'protein': 20, 'carbs': 6, 'fat': 4},
    'seaweed_salad': {'calories': 70, 'protein': 2, 'carbs': 10, 'fat': 2},
    'shrimp_and_grits': {'calories': 350, 'protein': 18, 'carbs': 30, 'fat': 18},
    'spaghetti_bolognese': {'calories': 400, 'protein': 20, 'carbs': 48, 'fat': 14},
    'spaghetti_carbonara': {'calories': 450, 'protein': 18, 'carbs': 48, 'fat': 20},
    'spring_rolls': {'calories': 200, 'protein': 5, 'carbs': 25, 'fat': 9},
    'steak': {'calories': 300, 'protein': 30, 'carbs': 0, 'fat': 20},
    'strawberry_shortcake': {'calories': 280, 'protein': 3, 'carbs': 38, 'fat': 13},
    'sushi': {'calories': 200, 'protein': 8, 'carbs': 30, 'fat': 5},
    'tacos': {'calories': 280, 'protein': 14, 'carbs': 22, 'fat': 15},
    'takoyaki': {'calories': 250, 'protein': 10, 'carbs': 28, 'fat': 12},
    'tiramisu': {'calories': 300, 'protein': 5, 'carbs': 32, 'fat': 16},
    'tuna_tartare': {'calories': 160, 'protein': 22, 'carbs': 3, 'fat': 7},
    'waffles': {'calories': 290, 'protein': 7, 'carbs': 38, 'fat': 12},

    # === Indian & Western 35 Varieties (additional) ===
    'baked_potato': {'calories': 160, 'protein': 4, 'carbs': 36, 'fat': 0},
    'butter_naan': {'calories': 310, 'protein': 8, 'carbs': 45, 'fat': 12},
    'chai': {'calories': 120, 'protein': 3, 'carbs': 18, 'fat': 4},
    'chapati': {'calories': 120, 'protein': 3, 'carbs': 25, 'fat': 1},
    'chole_bhature': {'calories': 450, 'protein': 12, 'carbs': 52, 'fat': 22},
    'crispy_chicken': {'calories': 350, 'protein': 25, 'carbs': 18, 'fat': 20},
    'dal_makhani': {'calories': 230, 'protein': 9, 'carbs': 28, 'fat': 9},
    'dhokla': {'calories': 130, 'protein': 5, 'carbs': 20, 'fat': 3},
    'donut': {'calories': 280, 'protein': 4, 'carbs': 33, 'fat': 15},
    'dosa': {'calories': 170, 'protein': 4, 'carbs': 28, 'fat': 5},
    'fries': {'calories': 312, 'protein': 3, 'carbs': 41, 'fat': 15},
    'fried_rice_indian': {'calories': 350, 'protein': 8, 'carbs': 50, 'fat': 14},
    'idli': {'calories': 80, 'protein': 2, 'carbs': 16, 'fat': 0},
    'jalebi': {'calories': 380, 'protein': 2, 'carbs': 58, 'fat': 16},
    'kaathi_rolls': {'calories': 320, 'protein': 14, 'carbs': 32, 'fat': 14},
    'kadai_paneer': {'calories': 280, 'protein': 14, 'carbs': 10, 'fat': 20},
    'kulfi': {'calories': 200, 'protein': 4, 'carbs': 28, 'fat': 8},
    'masala_dosa': {'calories': 250, 'protein': 6, 'carbs': 35, 'fat': 10},
    'momos': {'calories': 230, 'protein': 10, 'carbs': 28, 'fat': 8},
    'naan': {'calories': 260, 'protein': 7, 'carbs': 40, 'fat': 8},
    'paani_puri': {'calories': 150, 'protein': 3, 'carbs': 22, 'fat': 6},
    'pakode': {'calories': 300, 'protein': 5, 'carbs': 25, 'fat': 20},
    'pav_bhaji': {'calories': 350, 'protein': 8, 'carbs': 42, 'fat': 16},
    'sandwich': {'calories': 350, 'protein': 15, 'carbs': 35, 'fat': 16},
    'taco': {'calories': 280, 'protein': 14, 'carbs': 22, 'fat': 15},
    'taquito': {'calories': 220, 'protein': 8, 'carbs': 24, 'fat': 10},
    'burger': {'calories': 400, 'protein': 22, 'carbs': 30, 'fat': 22},
}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/classify', methods=['POST'])
def classify_image():
    if model is None:
        return jsonify({'error': 'No model loaded. Train a model first.'}), 500

    try:
        file = request.files.get('image')
        if not file:
            return jsonify({'error': 'No image provided'}), 400

        img = Image.open(file.stream).convert('RGB')
        img = img.resize(TARGET_SIZE)
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_idx = np.argmax(prediction)
        predicted_class = CATEGORIES[predicted_idx]
        confidence = float(prediction[0][predicted_idx])

        nutrient_info = NUTRIENTS.get(predicted_class, {
            'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0
        })

        return jsonify({
            'class': predicted_class,
            'confidence': round(confidence * 100, 1),
            'nutrients': nutrient_info
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
