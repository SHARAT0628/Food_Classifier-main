# ============================================================
# 🍕 NutriScan — Food Classifier Training (KAGGLE NOTEBOOK)
# ============================================================
# Steps:
#   1. Go to kaggle.com → + Create → New Notebook
#   2. Settings (right panel) → Accelerator → GPU T4 x2
#   3. + Add Input → search & add these 2 datasets:
#      - "food-101" by dansbecker
#      - "food image classification dataset" by harishkumardatalab
#   4. Copy each CELL below into separate notebook cells
#   5. Run all cells
# ============================================================


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 1: Check datasets & find paths                    ║
# ╚══════════════════════════════════════════════════════════╝

import os

print("📂 Datasets in /kaggle/input/:")
for d in os.listdir('/kaggle/input/'):
    print(f"  📁 {d}/")

# Auto-find Food-101
food101_path = None
for root, dirs, files in os.walk('/kaggle/input/'):
    if len(dirs) > 90 and any('apple_pie' in d for d in dirs):
        food101_path = root
        break

# Auto-find Food-35
food35_path = None
for root, dirs, files in os.walk('/kaggle/input/'):
    if 5 < len(dirs) < 50 and any('naan' in d.lower() or 'Naan' in d for d in dirs):
        food35_path = root
        break

# Fallback: manual search
if not food101_path:
    for root, dirs, files in os.walk('/kaggle/input/'):
        if len(dirs) > 90:
            food101_path = root
            break

if not food35_path:
    for root, dirs, files in os.walk('/kaggle/input/'):
        if 20 < len(dirs) < 50 and root != food101_path:
            food35_path = root
            break

print(f"\n🔍 Food-101: {food101_path}")
if food101_path:
    cats = [d for d in os.listdir(food101_path) if os.path.isdir(os.path.join(food101_path, d))]
    print(f"   → {len(cats)} categories")

print(f"🔍 Food-35:  {food35_path}")
if food35_path:
    cats = [d for d in os.listdir(food35_path) if os.path.isdir(os.path.join(food35_path, d))]
    print(f"   → {len(cats)} categories: {cats[:8]}...")


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 2: Build combined dataset on disk                 ║
# ╚══════════════════════════════════════════════════════════╝

import os, shutil
from PIL import Image

OUTPUT = '/kaggle/working/combined_dataset'
TARGET_SIZE = (224, 224)
MAX_PER_CLASS = 400

# Copy Food-101
if food101_path:
    print("📁 Copying Food-101 images...")
    for i, cat in enumerate(sorted(os.listdir(food101_path))):
        src = os.path.join(food101_path, cat)
        if not os.path.isdir(src):
            continue
        dest = os.path.join(OUTPUT, cat.lower().replace(' ', '_'))
        os.makedirs(dest, exist_ok=True)
        imgs = [f for f in os.listdir(src) if f.lower().endswith(('.jpg','.jpeg','.png'))]
        for j, img in enumerate(imgs[:MAX_PER_CLASS]):
            try:
                im = Image.open(os.path.join(src, img)).convert('RGB').resize(TARGET_SIZE)
                im.save(os.path.join(dest, f'{j}.jpg'))
            except:
                continue
        if (i+1) % 20 == 0:
            print(f"  [{i+1}] {cat}: {min(len(imgs), MAX_PER_CLASS)} imgs")

# Copy Food-35
if food35_path:
    print("\n📁 Adding Food-35 images...")
    for cat in sorted(os.listdir(food35_path)):
        src = os.path.join(food35_path, cat)
        if not os.path.isdir(src):
            continue
        norm = cat.lower().replace(' ', '_')
        dest = os.path.join(OUTPUT, norm)
        os.makedirs(dest, exist_ok=True)
        existing = len(os.listdir(dest))
        imgs = [f for f in os.listdir(src) if f.lower().endswith(('.jpg','.jpeg','.png','.webp'))]
        for j, img in enumerate(imgs[:MAX_PER_CLASS]):
            try:
                im = Image.open(os.path.join(src, img)).convert('RGB').resize(TARGET_SIZE)
                im.save(os.path.join(dest, f'f35_{j}.jpg'))
            except:
                continue
        print(f"  {norm}: +{min(len(imgs), MAX_PER_CLASS)} imgs")

total_cats = len(os.listdir(OUTPUT))
total_imgs = sum(len(os.listdir(os.path.join(OUTPUT, d))) for d in os.listdir(OUTPUT))
print(f"\n✅ Combined: {total_cats} categories, {total_imgs} images")


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 3: TRAIN (with checkpoint after Phase 1!)         ║
# ╚══════════════════════════════════════════════════════════╝

import os, json
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

DATASET_PATH = '/kaggle/working/combined_dataset'
MODEL_DIR = '/kaggle/working/model'
os.makedirs(MODEL_DIR, exist_ok=True)

# Data generators
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20, width_shift_range=0.2,
    height_shift_range=0.2, shear_range=0.2,
    zoom_range=0.2, horizontal_flip=True,
    fill_mode='nearest', validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    DATASET_PATH, target_size=(224, 224),
    batch_size=32, class_mode='sparse', subset='training'
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input, validation_split=0.2
)
val_gen = val_datagen.flow_from_directory(
    DATASET_PATH, target_size=(224, 224),
    batch_size=32, class_mode='sparse', subset='validation'
)

num_classes = len(train_gen.class_indices)
print(f"[✓] {num_classes} classes | Train: {train_gen.samples} | Val: {val_gen.samples}")

# Build model
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# ── Phase 1 ──
print("\n── Phase 1: Transfer Learning ──────────────────────────")
model.fit(
    train_gen, epochs=15,
    validation_data=val_gen,
    callbacks=[
        EarlyStopping(patience=4, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=2, verbose=1),
        ModelCheckpoint(f'{MODEL_DIR}/checkpoint_phase1.keras',
                        save_best_only=True, verbose=1),  # SAVES AFTER EACH BEST EPOCH!
    ]
)
print("✅ Phase 1 complete — checkpoint saved!")

# ── Phase 2: Fine-tune ──
print("\n── Phase 2: Fine-tuning ────────────────────────────────")
for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(optimizer=Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(
    train_gen, epochs=10,
    validation_data=val_gen,
    callbacks=[
        EarlyStopping(patience=3, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=2, verbose=1),
        ModelCheckpoint(f'{MODEL_DIR}/food_classifier_combined.keras',
                        save_best_only=True, verbose=1),
    ]
)

# Save final model + categories
scores = model.evaluate(val_gen, verbose=0)
categories_list = list(train_gen.class_indices.keys())

model.save(f'{MODEL_DIR}/food_classifier_combined.keras')
with open(f'{MODEL_DIR}/categories.json', 'w') as f:
    json.dump(categories_list, f, indent=2)

print(f"\n{'='*50}")
print(f"  ✅ Training Complete!")
print(f"  📊 Val Accuracy: {scores[1]*100:.1f}%")
print(f"  🍕 Categories:   {len(categories_list)}")
print(f"{'='*50}")
print(f"[✓] Model saved to {MODEL_DIR}/")


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 4: Download model                                 ║
# ╚══════════════════════════════════════════════════════════╝

# On Kaggle, go to the right panel → Output section
# Your files will be in /kaggle/working/model/
# Click "Save Version" (top right) → "Quick Save"
# Then download from the Output tab

import shutil
# Also copy to a convenient location
shutil.copy(f'{MODEL_DIR}/food_classifier_combined.keras', '/kaggle/working/food_classifier_combined.keras')
shutil.copy(f'{MODEL_DIR}/categories.json', '/kaggle/working/categories.json')
print("✅ Files ready in /kaggle/working/ — use 'Save Version' to download!")
print("   Download: food_classifier_combined.keras + categories.json")
