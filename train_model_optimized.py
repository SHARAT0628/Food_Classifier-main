import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from keras._tf_keras.keras.applications import EfficientNetB0
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras._tf_keras.keras.utils import Sequence
import json
import pickle
import random

FOOD101_PATH = 'food-101/images'
FOOD35_PATH = 'food-35'
TARGET_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
MODEL_SAVE_PATH = 'model/food_classifier_combined.keras'
CATEGORIES_SAVE_PATH = 'model/categories.json'

def discover_categories():
    categories = set()
    if os.path.exists(FOOD101_PATH):
        for cat in os.listdir(FOOD101_PATH):
            if os.path.isdir(os.path.join(FOOD101_PATH, cat)):
                categories.add(cat.lower().replace(' ', '_'))
    if os.path.exists(FOOD35_PATH):
        for cat in os.listdir(FOOD35_PATH):
            if os.path.isdir(os.path.join(FOOD35_PATH, cat)):
                categories.add(cat.lower().replace(' ', '_'))
    return sorted(list(categories))

def get_image_paths(categories, max_per_category=700):
    cat_to_idx = {cat: idx for idx, cat in enumerate(categories)}
    image_paths = []
    labels = []
    datasets = []

    if os.path.exists(FOOD101_PATH):
        for folder in os.listdir(FOOD101_PATH):
            if os.path.isdir(os.path.join(FOOD101_PATH, folder)):
                datasets.append((os.path.join(FOOD101_PATH, folder), folder.lower().replace(' ', '_')))

    if os.path.exists(FOOD35_PATH):
        for folder in os.listdir(FOOD35_PATH):
            if os.path.isdir(os.path.join(FOOD35_PATH, folder)):
                datasets.append((os.path.join(FOOD35_PATH, folder), folder.lower().replace(' ', '_')))

    for folder_path, cat_name in datasets:
        if cat_name not in cat_to_idx: continue
        cat_idx = cat_to_idx[cat_name]
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        random.shuffle(files)
        for f in files[:max_per_category]:
            image_paths.append(f)
            labels.append(cat_idx)

    return image_paths, labels

class FoodDataGenerator(Sequence):
    def __init__(self, image_paths, labels, batch_size, target_size, augment=False, shuffle=True):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.augment = augment
        if self.augment:
            self.datagen = ImageDataGenerator(
                rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
                shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest'
            )
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_paths = [self.image_paths[k] for k in indexes]
        batch_labels = [self.labels[k] for k in indexes]
        
        X = []
        y = []

        for path, label in zip(batch_paths, batch_labels):
            try:
                img = Image.open(path).convert('RGB').resize(self.target_size)
                img_arr = np.array(img, dtype=np.float32)
                if self.augment:
                    img_arr = self.datagen.random_transform(img_arr)
                X.append(img_arr)
                y.append(label)
            except:
                pass
                
        if len(X) == 0:
            return np.zeros((1, *self.target_size, 3), dtype=np.float32), np.zeros((1), dtype=np.int32)
            
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

def build_model(num_classes):
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers: layer.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model, base_model

def main():
    print("============================================================")
    print("  Food Classifier Training - Streaming Dataset")
    print("============================================================")

    categories = discover_categories()
    if not categories: return

    print("\n[INFO] Finding image paths to build data loaders...")
    paths, labels = get_image_paths(categories, max_per_category=700) # Limited to save time/space
    print(f"[OK] Total valid images discovered: {len(paths)}")

    train_paths, val_paths, train_labels, val_labels = train_test_split(paths, labels, test_size=0.2, random_state=42, stratify=labels)
    
    train_gen = FoodDataGenerator(train_paths, train_labels, BATCH_SIZE, TARGET_SIZE, augment=True)
    val_gen = FoodDataGenerator(val_paths, val_labels, BATCH_SIZE, TARGET_SIZE, augment=False)

    print("\n-- Phase 1: Transfer Learning --------------------------")
    if os.path.exists(MODEL_SAVE_PATH):
        from keras._tf_keras.keras.models import load_model
        print(f"\n[INFO] Found existing model at {MODEL_SAVE_PATH}, resuming training!")
        model = load_model(MODEL_SAVE_PATH)
        base_model = model
    else:
        model, base_model = build_model(len(categories))
    
    os.makedirs('model', exist_ok=True)
    callbacks = [
        EarlyStopping(patience=3, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=2, verbose=1),
        ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, verbose=1)
    ]

    history = model.fit(train_gen, epochs=EPOCHS, validation_data=val_gen, callbacks=callbacks)

    print("\n-- Phase 2: Fine-tuning ---------------------------------")
    for layer in base_model.layers[-20:]: layer.trainable = True
    from keras._tf_keras.keras.optimizers import Adam
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    history2 = model.fit(train_gen, epochs=10, validation_data=val_gen, callbacks=callbacks)

    print(f"\n[OK] Model saved to: {MODEL_SAVE_PATH}")
    with open(CATEGORIES_SAVE_PATH, 'w') as f: json.dump(categories, f, indent=2)

if __name__ == '__main__':
    main()
