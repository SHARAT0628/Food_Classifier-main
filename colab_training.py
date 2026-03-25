# ============================================================
# 🍕 NutriScan — Food Classifier Training (Google Colab)
# ============================================================
# Copy each section below into a SEPARATE cell in Google Colab
# Make sure to set Runtime → Change runtime type → T4 GPU
# ============================================================


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 1: Mount Google Drive                             ║
# ╚══════════════════════════════════════════════════════════╝

from google.colab import drive
drive.mount('/content/drive')


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 2: Download Datasets from Kaggle                  ║
# ║  (You need a Kaggle API key — see instructions below)   ║
# ╚══════════════════════════════════════════════════════════╝

# --- Option A: Download directly from Kaggle API ---
# 1. Go to kaggle.com → Your Profile → Settings → Create New API Token
# 2. This downloads a kaggle.json file
# 3. Upload it when prompted below:

import os
from google.colab import files

# Upload your kaggle.json
print("📂 Upload your kaggle.json file (from Kaggle → Settings → API):")
uploaded = files.upload()

# Set up Kaggle credentials
os.makedirs('/root/.kaggle', exist_ok=True)
os.rename('kaggle.json', '/root/.kaggle/kaggle.json')
os.chmod('/root/.kaggle/kaggle.json', 0o600)

# Download datasets
print("\n📥 Downloading Food-101 dataset...")
os.system('kaggle datasets download -d dansbecker/food-101 -p /content/')
print("📥 Downloading 35 Varieties dataset...")
os.system('kaggle datasets download -d harishkumardatalab/food-image-classification-dataset -p /content/')

# Extract
print("\n📦 Extracting datasets...")
os.system('unzip -q /content/food-101.zip -d /content/food-101-raw')
os.system('unzip -q /content/food-image-classification-dataset.zip -d /content/food-35-raw')

print("✅ Datasets downloaded and extracted!")


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 2B (Alternative): If you already uploaded         ║
# ║  datasets to Google Drive, use this instead of Cell 2   ║
# ╚══════════════════════════════════════════════════════════╝

# Skip Cell 2 and run this instead if datasets are on your Drive:
# import os
# print("📦 Copying from Google Drive...")
# os.system('cp -r "/content/drive/MyDrive/food-101" /content/food-101-raw')
# os.system('cp -r "/content/drive/MyDrive/food-image-classification-dataset" /content/food-35-raw')
# print("✅ Done!")


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 3: Check dataset structure & set paths            ║
# ╚══════════════════════════════════════════════════════════╝

import os

# Auto-detect Food-101 images path
food101_path = None
for root, dirs, fls in os.walk('/content/food-101-raw'):
    if len(dirs) > 50:  # Food-101 has 101 subdirectories
        food101_path = root
        break

# Auto-detect Food-35 images path
food35_path = None
for root, dirs, fls in os.walk('/content/food-35-raw'):
    if len(dirs) > 10 and len(dirs) <= 40:
        food35_path = root
        break

print(f"🔍 Food-101 path: {food101_path}")
if food101_path:
    cats = [d for d in os.listdir(food101_path) if os.path.isdir(os.path.join(food101_path, d))]
    print(f"   → {len(cats)} categories found")

print(f"\n🔍 Food-35 path:  {food35_path}")
if food35_path:
    cats = [d for d in os.listdir(food35_path) if os.path.isdir(os.path.join(food35_path, d))]
    print(f"   → {len(cats)} categories found")
    print(f"   → Categories: {sorted(cats)}")


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 4: Train the Model                                ║
# ╚══════════════════════════════════════════════════════════╝

import os
import numpy as np
import json
import pickle
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# ── Config ──
TARGET_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15      # Phase 1 epochs
FT_EPOCHS = 10   # Phase 2 (fine-tuning) epochs
MAX_PER_CATEGORY = 800  # Limit images per category to control memory


# Step 1: Discover categories
print("=" * 60)
print("  🍕 NutriScan Model Training")
print("=" * 60)

categories = set()

# Food-101
if food101_path:
    for cat in os.listdir(food101_path):
        if os.path.isdir(os.path.join(food101_path, cat)):
            categories.add(cat.lower().replace(' ', '_'))
    count = len([c for c in os.listdir(food101_path) if os.path.isdir(os.path.join(food101_path, c))])
    print(f"[✓] Food-101: {count} categories")

# Food-35
if food35_path:
    for cat in os.listdir(food35_path):
        if os.path.isdir(os.path.join(food35_path, cat)):
            categories.add(cat.lower().replace(' ', '_'))
    count = len([c for c in os.listdir(food35_path) if os.path.isdir(os.path.join(food35_path, c))])
    print(f"[✓] Food-35:  {count} categories")

categories = sorted(list(categories))
print(f"\n[✓] Total unique categories: {len(categories)}")
cat_to_idx = {cat: idx for idx, cat in enumerate(categories)}


# Step 2: Load images
print("\n── Loading images ──────────────────────────────────────")
images = []
labels = []
datasets = []

if food101_path:
    for folder in os.listdir(food101_path):
        fp = os.path.join(food101_path, folder)
        if os.path.isdir(fp):
            norm = folder.lower().replace(' ', '_')
            if norm in cat_to_idx:
                datasets.append((fp, norm))

if food35_path:
    for folder in os.listdir(food35_path):
        fp = os.path.join(food35_path, folder)
        if os.path.isdir(fp):
            norm = folder.lower().replace(' ', '_')
            if norm in cat_to_idx:
                datasets.append((fp, norm))

total = len(datasets)
for i, (folder_path, cat_name) in enumerate(datasets):
    cat_idx = cat_to_idx[cat_name]
    img_files = [f for f in os.listdir(folder_path)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    count = 0
    for img_file in img_files[:MAX_PER_CATEGORY]:
        try:
            img = Image.open(os.path.join(folder_path, img_file)).convert('RGB')
            img = img.resize(TARGET_SIZE)
            images.append(np.array(img))
            labels.append(cat_idx)
            count += 1
        except:
            continue
    if (i + 1) % 20 == 0 or i == total - 1:
        print(f"  [{i+1}/{total}] Loaded {cat_name} ({count} imgs)")

X = np.array(images)
y = np.array(labels)
del images, labels  # Free memory
print(f"\n[✓] Total images: {len(X)}")


# Step 3: Preprocess & split
X = X / 255.0
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"[✓] Train: {len(X_train)} | Val: {len(X_val)}")

datagen = ImageDataGenerator(
    rotation_range=20, width_shift_range=0.2,
    height_shift_range=0.2, shear_range=0.2,
    zoom_range=0.2, horizontal_flip=True, fill_mode='nearest'
)


# Step 4: Build model
print("\n── Phase 1: Transfer Learning ──────────────────────────")
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(len(categories), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print(f"[✓] Model built: EfficientNetB0 → {len(categories)} classes")

callbacks_p1 = [
    EarlyStopping(patience=3, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(factor=0.5, patience=2, verbose=1),
]

history_p1 = model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val, y_val),
    callbacks=callbacks_p1
)


# Step 5: Fine-tune
print("\n── Phase 2: Fine-tuning ────────────────────────────────")
for layer in base_model.layers[-20:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

os.makedirs('/content/model', exist_ok=True)

callbacks_p2 = [
    EarlyStopping(patience=3, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(factor=0.5, patience=2, verbose=1),
    ModelCheckpoint('/content/model/food_classifier_combined.keras',
                    save_best_only=True, verbose=1),
]

history_p2 = model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    epochs=FT_EPOCHS,
    validation_data=(X_val, y_val),
    callbacks=callbacks_p2
)


# Step 6: Final evaluation
print("\n── Results ─────────────────────────────────────────────")
scores = model.evaluate(X_val, y_val, verbose=0)
print(f"[✓] Validation Loss:     {scores[0]:.4f}")
print(f"[✓] Validation Accuracy: {scores[1]:.4f}")

# Save model & categories
model.save('/content/model/food_classifier_combined.keras')
with open('/content/model/categories.json', 'w') as f:
    json.dump(categories, f, indent=2)

print(f"\n✅ Training complete! Model saved with {len(categories)} categories.")


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 5: Download the trained model to your PC          ║
# ╚══════════════════════════════════════════════════════════╝

from google.colab import files

print("📥 Downloading model files...")
files.download('/content/model/food_classifier_combined.keras')
files.download('/content/model/categories.json')
print("✅ Download started! Save both files to your project's model/ folder.")


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 5B (Alternative): Save to Google Drive instead    ║
# ╚══════════════════════════════════════════════════════════╝

# import shutil
# drive_path = '/content/drive/MyDrive/NutriScan_Model/'
# os.makedirs(drive_path, exist_ok=True)
# shutil.copy('/content/model/food_classifier_combined.keras', drive_path)
# shutil.copy('/content/model/categories.json', drive_path)
# print(f"✅ Model saved to Google Drive: {drive_path}")
