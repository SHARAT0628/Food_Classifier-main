"""
Training script for Food Classifier — combines Food-101 + 35 Indian/Western varieties.

Usage:
    1. Download Food-101 from: https://www.kaggle.com/datasets/dansbecker/food-101
    2. Download 35 Varieties from: https://www.kaggle.com/datasets/harishkumardatalab/food-image-classification-dataset
    3. Extract them into the project folder:
       - food-101/images/<category>/  (101 categories)
       - food-35/<category>/          (35 categories)
    4. Run: python train_model.py
"""

import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from keras._tf_keras.keras.applications import EfficientNetB0
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import json
import pickle

# ── Configuration ──────────────────────────────────────────────────────
FOOD101_PATH = 'food-101/images'       # Food-101 dataset path
FOOD35_PATH = 'food-35'                # 35 varieties dataset path
TARGET_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
MODEL_SAVE_PATH = 'model/food_classifier_combined.keras'
CATEGORIES_SAVE_PATH = 'model/categories.json'


def discover_categories():
    """Discover all unique categories from both datasets."""
    categories = set()

    # Food-101 categories
    if os.path.exists(FOOD101_PATH):
        for cat in os.listdir(FOOD101_PATH):
            if os.path.isdir(os.path.join(FOOD101_PATH, cat)):
                categories.add(cat.lower().replace(' ', '_'))
        print(f"[✓] Food-101: Found {len([c for c in os.listdir(FOOD101_PATH) if os.path.isdir(os.path.join(FOOD101_PATH, c))])} categories")
    else:
        print(f"[!] Food-101 not found at '{FOOD101_PATH}' — skipping")

    # Food-35 categories
    if os.path.exists(FOOD35_PATH):
        for cat in os.listdir(FOOD35_PATH):
            cat_path = os.path.join(FOOD35_PATH, cat)
            if os.path.isdir(cat_path):
                categories.add(cat.lower().replace(' ', '_'))
        print(f"[✓] Food-35:  Found {len([c for c in os.listdir(FOOD35_PATH) if os.path.isdir(os.path.join(FOOD35_PATH, c))])} categories")
    else:
        print(f"[!] Food-35 not found at '{FOOD35_PATH}' — skipping")

    categories = sorted(list(categories))
    print(f"\n[✓] Total unique categories: {len(categories)}")
    return categories


def load_images(categories, max_per_category=1000):
    """Load images from both datasets."""
    images = []
    labels = []
    cat_to_idx = {cat: idx for idx, cat in enumerate(categories)}

    # Map folders to normalized category names for each dataset
    datasets = []

    if os.path.exists(FOOD101_PATH):
        for folder in os.listdir(FOOD101_PATH):
            folder_path = os.path.join(FOOD101_PATH, folder)
            if os.path.isdir(folder_path):
                norm_name = folder.lower().replace(' ', '_')
                if norm_name in cat_to_idx:
                    datasets.append((folder_path, norm_name))

    if os.path.exists(FOOD35_PATH):
        for folder in os.listdir(FOOD35_PATH):
            folder_path = os.path.join(FOOD35_PATH, folder)
            if os.path.isdir(folder_path):
                norm_name = folder.lower().replace(' ', '_')
                if norm_name in cat_to_idx:
                    datasets.append((folder_path, norm_name))

    total = len(datasets)
    for i, (folder_path, cat_name) in enumerate(datasets):
        cat_idx = cat_to_idx[cat_name]
        img_files = [f for f in os.listdir(folder_path)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.avif'))]
        count = 0

        for img_file in img_files[:max_per_category]:
            try:
                img_path = os.path.join(folder_path, img_file)
                img = Image.open(img_path).convert('RGB')
                img = img.resize(TARGET_SIZE)
                images.append(np.array(img))
                labels.append(cat_idx)
                count += 1
            except Exception as e:
                continue

        progress = f"[{i+1}/{total}]"
        print(f"  {progress} {cat_name}: {count} images loaded")

    return np.array(images), np.array(labels)


def build_model(num_classes):
    """Build EfficientNetB0 model with transfer learning."""
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print(f"\n[✓] Model built: EfficientNetB0 + Dense layers → {num_classes} classes")
    return model, base_model


def fine_tune(model, base_model, datagen, X_train, y_train, X_val, y_val):
    """Fine-tune the last 20 layers of the base model."""
    print("\n── Phase 2: Fine-tuning ─────────────────────────────────")

    # Unfreeze the last 20 layers
    for layer in base_model.layers[-20:]:
        layer.trainable = True

    from keras._tf_keras.keras.optimizers import Adam
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    callbacks = [
        EarlyStopping(patience=3, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=2, verbose=1),
        ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, verbose=1)
    ]

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        epochs=10,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )
    return history


def main():
    print("=" * 60)
    print("  Food Classifier Training — Combined Dataset")
    print("=" * 60)

    # Step 1: Discover categories
    print("\n── Discovering categories ───────────────────────────────")
    categories = discover_categories()

    if len(categories) == 0:
        print("\n[✗] No datasets found! Please extract datasets as:")
        print(f"    {FOOD101_PATH}/<category>/<images>")
        print(f"    {FOOD35_PATH}/<category>/<images>")
        return

    # Step 2: Load images
    print("\n── Loading images ──────────────────────────────────────")
    X, y = load_images(categories)
    print(f"\n[✓] Total images loaded: {len(X)}")

    # Step 3: Preprocess
    X = X / 255.0
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"[✓] Train: {len(X_train)} | Validation: {len(X_val)}")

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Step 4: Build and train model
    print("\n── Phase 1: Transfer Learning ──────────────────────────")
    model, base_model = build_model(len(categories))

    callbacks = [
        EarlyStopping(patience=3, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=2, verbose=1),
    ]

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )

    # Step 5: Fine-tune
    fine_tune(model, base_model, datagen, X_train, y_train, X_val, y_val)

    # Step 6: Evaluate
    print("\n── Evaluation ─────────────────────────────────────────")
    scores = model.evaluate(X_val, y_val, verbose=0)
    print(f"[✓] Validation Loss:     {scores[0]:.4f}")
    print(f"[✓] Validation Accuracy: {scores[1]:.4f}")

    # Step 7: Save
    os.makedirs('model', exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print(f"\n[✓] Model saved to: {MODEL_SAVE_PATH}")

    # Save categories list
    with open(CATEGORIES_SAVE_PATH, 'w') as f:
        json.dump(categories, f, indent=2)
    print(f"[✓] Categories saved to: {CATEGORIES_SAVE_PATH}")

    # Save training history
    with open('training_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)

    print(f"\n{'=' * 60}")
    print(f"  Training complete! {len(categories)} food categories.")
    print(f"  Run 'python app.py' to start the web app.")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
