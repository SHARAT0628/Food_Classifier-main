from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False
from keras._tf_keras.keras.models import load_model
import os
import json
import cv2
import traceback

app = Flask(__name__)

# -- Constants --
STANDARD_PLATE_DIAMETER_CM = 25.4 
DEFAULT_DENSITY = 0.8
DEFAULT_THICKNESS = 2.0
TARGET_SIZE = (224, 224)

FOODSEG103_NAMES = [
    "candy", "egg_tart", "french_fries", "chocolate", "biscuit", "popcorn", "pudding", "ice_cream", 
    "cheese_butter", "cake", "wine", "milkshake", "coffee", "juice", "milk", "tea", "almond", 
    "red_beans", "cashew", "dried_cranberries", "soy", "walnut", "peanut", "egg", "apple", 
    "date", "apricot", "avocado", "banana", "strawberry", "cherry", "blueberry", "raspberry", 
    "mango", "olives", "peach", "lemon", "pear", "fig", "pineapple", "grape", "kiwi", "melon", 
    "orange", "watermelon", "steak", "pork", "chicken_duck", "sausage", "fried_meat", "lamb", 
    "sauce", "crab", "fish", "shellfish", "shrimp", "soup", "bread", "corn", "hamburg", "pizza", 
    "hanamaki_baozi", "wonton_dumplings", "pasta", "noodles", "rice", "pie", "tofu", "eggplant", 
    "potato", "garlic", "cauliflower", "tomato", "kelp", "seaweed", "spring_onion", "rape", 
    "ginger", "okra", "lettuce", "pumpkin", "cucumber", "white_radish", "carrot", "asparagus", 
    "bamboo_shoots", "broccoli", "celery", "spinach", "green_pepper", "red_pepper", "dried_mushroom", 
    "shiitake", "needle_mushroom", "sweet_potato", "kohlrabi", "chili", "enoki_mushroom", 
    "oyster_mushroom", "white_button_mushroom", "salad", "other_ingredients"
]

MODEL_YOLO_PATH = 'model/best.pt'
MODEL_KERAS_PATH = 'model/food_classifier_combined.keras'
CATEGORIES_PATH = 'model/categories.json'

yolo_model = None
keras_model = None
KERAS_CATEGORIES = []

if HAS_YOLO and os.path.exists(MODEL_YOLO_PATH):
    try: yolo_model = YOLO(MODEL_YOLO_PATH)
    except: pass

if os.path.exists(MODEL_KERAS_PATH) and os.path.exists(CATEGORIES_PATH):
    try:
        keras_model = load_model(MODEL_KERAS_PATH)
        with open(CATEGORIES_PATH, 'r') as f: KERAS_CATEGORIES = json.load(f)
    except: pass

NUTRIENTS = {
    'soup': {'calories': 50, 'protein': 2, 'carbs': 8, 'fat': 1.5, 'serving_size': 200, 'density': 1.0, 'thickness': 4.0},
    'sauce': {'calories': 100, 'protein': 1, 'carbs': 12, 'fat': 6, 'serving_size': 50, 'density': 1.1, 'thickness': 1.5},
    'potato': {'calories': 77, 'protein': 2, 'carbs': 17, 'fat': 0.1, 'serving_size': 150, 'density': 0.7, 'thickness': 3.0},
    'rice': {'calories': 130, 'protein': 2.7, 'carbs': 28, 'fat': 0.3, 'serving_size': 100, 'density': 0.9, 'thickness': 2.0},
    'bread': {'calories': 265, 'protein': 9, 'carbs': 49, 'fat': 3.2, 'serving_size': 100, 'density': 0.4, 'thickness': 2.0},
    'cake': {'calories': 350, 'protein': 4, 'carbs': 50, 'fat': 16, 'serving_size': 100, 'density': 0.7, 'thickness': 5.0},
    'other_ingredients': {'calories': 50, 'protein': 2, 'carbs': 5, 'fat': 2, 'serving_size': 100, 'density': 0.8, 'thickness': 2.0},
}

INDIAN_COMPATIBILITY = {
    'idli': {'calories': 80, 'protein': 2, 'carbs': 16, 'fat': 0.2, 'serving_size': 60, 'density': 0.6, 'thickness': 2.5},
    'sambar': {'calories': 60, 'protein': 3, 'carbs': 10, 'fat': 2, 'serving_size': 150, 'density': 1.0, 'thickness': 5.0},
    'chutney': {'calories': 80, 'protein': 1, 'carbs': 6, 'fat': 6, 'serving_size': 30, 'density': 1.1, 'thickness': 1.5},
    'biryani': {'calories': 450, 'protein': 20, 'carbs': 55, 'fat': 18, 'serving_size': 350, 'density': 0.85, 'thickness': 3.5},
}

def detect_plates(img_pil):
    try:
        w, h = img_pil.size
        scale = 800 / float(w) if w > 800 else 1.0
        img_np = np.array(img_pil.resize((int(w*scale), int(h*scale))))
        gray = cv2.cvtColor(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(cv2.medianBlur(gray, 5), cv2.HOUGH_GRADIENT, 1.2, 100, param1=50, param2=30, minRadius=int(40*scale), maxRadius=int(400*scale))
        plates = []
        if circles is not None:
            for i in np.uint16(np.around(circles))[0, :]:
                orad = int(i[2]/scale)
                plates.append({'center': (int(i[0]/scale), int(i[1]/scale)), 'radius': orad, 'scale': STANDARD_PLATE_DIAMETER_CM/(2*orad)})
        return plates
    except: return []

@app.route('/')
def index(): return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_image():
    if not yolo_model: return jsonify({'error': 'YOLO model not loaded.'}), 500
    try:
        file = request.files.get('image')
        img = Image.open(file.stream).convert('RGB')
        plates = detect_plates(img)
        results = yolo_model(img)
        detections = []
        for r in results:
            if not hasattr(r, 'boxes'): continue
            for i in range(len(r.boxes)):
                box, conf = r.boxes[i], float(r.boxes[i].conf[0])
                if conf < 0.05: continue
                
                cls_id = int(box.cls[0])
                label = FOODSEG103_NAMES[cls_id] if yolo_model.names[cls_id].startswith("food_") else yolo_model.names[cls_id]
                coords = box.xyxy[0].tolist()
                
                # Hybrid Logic: Try to refine label using Keras
                final_label = label
                if keras_model:
                    try:
                        crop = img.crop(coords).resize(TARGET_SIZE)
                        pred = keras_model.predict(np.expand_dims(np.array(crop), 0), verbose=0)
                        k_idx = np.argmax(pred)
                        if pred[0][k_idx] > 0.35:
                            k_label = KERAS_CATEGORIES[k_idx].lower()
                            if (label == 'rice' and 'biryani' in k_label) or \
                               (label == 'bread' and ('dosa' in k_label or 'roti' in k_label)) or \
                               (label == 'cake' and 'idli' in k_label) or \
                               (label == 'potato' and 'idli' in k_label):
                                final_label = k_label
                    except: pass
                
                nutri = NUTRIENTS.get(final_label, INDIAN_COMPATIBILITY.get(final_label, NUTRIENTS['other_ingredients'])).copy()
                display_label = final_label.replace('_', ' ').title()
                
                # Sambar/Idli Synonym injection
                if 'soup' in final_label: display_label, nutri = "Soup (Sambar?)", INDIAN_COMPATIBILITY['sambar']
                elif 'sauce' in final_label: display_label, nutri = "Sauce (Chutney?)", INDIAN_COMPATIBILITY['chutney']
                elif 'potato' in final_label and conf < 0.3: display_label, nutri = "Sauce (Chutney?)", INDIAN_COMPATIBILITY['chutney']
                elif ('cake' in final_label or 'potato' in final_label) and 'idli' not in final_label: 
                    display_label, nutri = "Cake (Idly?)", INDIAN_COMPATIBILITY['idli']
                
                # Weight
                area_px = (coords[2]-coords[0]) * (coords[3]-coords[1]) * 0.75
                if hasattr(r, 'masks') and r.masks is not None and i < len(r.masks):
                    m = r.masks[i].data[0].cpu().numpy()
                    area_px = np.sum(m > 0.5) * (img.size[1]/m.shape[0]) * (img.size[0]/m.shape[1])
                
                weight = 100
                cx, cy = (coords[0]+coords[2])/2, (coords[1]+coords[3])/2
                plate = next((p for p in plates if np.sqrt((cx-p['center'][0])**2 + (cy-p['center'][1])**2) <= p['radius']*1.5), plates[0] if plates else None)
                if plate: weight = max(10, min(1500, round(area_px * (plate['scale']**2) * nutri.get('density', 0.8) * nutri.get('thickness', 2.0))))
                
                detections.append({'class': display_label, 'confidence': round(conf*100, 1), 'nutrients': nutri, 'detected_weight': weight})

        if not detections: return jsonify({'error': 'No food detected.'}), 404
        return jsonify({'is_multi': True, 'items': detections})
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__': app.run(debug=True)
