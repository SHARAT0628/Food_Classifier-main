from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display
import numpy as np
from sklearn.svm import SVC
from keras._tf_keras.keras.applications.vgg16 import VGG16
from keras._tf_keras.keras.models import Model, load_model
from keras._tf_keras.keras.layers import Dense, Flatten
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from tkinter import Tk, Label, Button, filedialog
import cv2
import pandas as pd
import pickle


DATASET_PATH = 'food11'
CATEGORIES = ['apple_pie', 'cheesecake', 'chicken_curry', 'french_fries', 'fried_rice',
              'hamburger', 'hot_dog', 'ice_cream', 'omelette', 'pizza', 'sushi']
SUBDIRECTORIES = ['train', 'test']
FOOD_CATEGORIES_PATH = 'Food Categories'
TARGET_SIZE = (224, 224)
NUTRIENTS = {'apple_pie': {'calories': 250, 'protein': 3, 'carbs': 40, 'fat': 10},
    'cheesecake': {'calories': 350, 'protein': 5, 'carbs': 30, 'fat': 20},
    'chicken_curry': {'calories': 400, 'protein': 20, 'carbs': 30, 'fat': 15},
    'french_fries': {'calories': 300, 'protein': 2, 'carbs': 50, 'fat': 15},
    'fried_rice': {'calories': 350, 'protein': 10, 'carbs': 45, 'fat': 8},
    'hamburger': {'calories': 500, 'protein': 25, 'carbs': 30, 'fat': 25},
    'hot_dog': {'calories': 350, 'protein': 12, 'carbs': 25, 'fat': 20},
    'ice_cream': {'calories': 300, 'protein': 4, 'carbs': 35, 'fat': 18},
    'omelette': {'calories': 300, 'protein': 15, 'carbs': 5, 'fat': 22},
    'pizza': {'calories': 400, 'protein': 10, 'carbs': 30, 'fat': 20},
    'sushi': {'calories': 250, 'protein': 8, 'carbs': 30, 'fat': 5}}


def create_datagen_train(s_x_train_preprocessed):
    # Define ImageDataGenerator
    s_datagen_train = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    s_datagen_train.fit(s_x_train_preprocessed)
    return s_datagen_train


# Train Model Again if required
def train_model(s_x_train_preprocessed, s_y_train, s_x_test_preprocessed, s_y_test):
    # Create datagen
    datagen_train = create_datagen_train(s_x_train_preprocessed)

    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = Flatten()(x)
    s_predictions = Dense(len(CATEGORIES), activation='softmax')(x)
    s_model = Model(inputs=base_model.input, outputs=s_predictions)
    for layer in base_model.layers:
        layer.trainable = False
    s_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    s_model.fit(datagen_train.flow(s_x_train_preprocessed, s_y_train, batch_size=32),
                steps_per_epoch=len(s_x_train_preprocessed) // 32,
                epochs=10,
                validation_data=(s_x_test_preprocessed, s_y_test))

    # Save the model
    s_model.save('food_classifier_1.keras')


def fine_tune_model(trained_model, s_datagen_train, s_x_train_preprocessed, s_y_train,
                    s_x_test_preprocessed, s_y_test, num_epochs=5):
    # Fine-tune the model
    for s_layer in model.layers[:10]:
        s_layer.trainable = True

    # Recompile the model
    trained_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(s_datagen_train.flow(s_x_train_preprocessed, s_y_train, batch_size=32),
                        steps_per_epoch=len(s_x_train_preprocessed) // 32,
                        epochs=num_epochs,
                        validation_data=(s_x_test_preprocessed, s_y_test))

    # Save the fine-tuned model
    model.save('fine_tuned_food_classifier.keras')
    with open('training_history.pkl', 'wb') as file :
        pickle.dump(history.history, file)


# test classification function
def test_classification(s_x_test_preprocessed, s_y_test):
    while True:
        selected_index = int(input("Enter the index of the test image you want to classify (-1 to exit): "))
        if selected_index == -1:
            print("Exiting...")
            break
        elif 0 <= selected_index - 1 < len(s_x_test_preprocessed):
            selected_image = s_x_test_preprocessed[selected_index - 1]
            selected_label = s_y_test[selected_index - 1]
            prediction = model.predict(np.expand_dims(selected_image, axis=0))
            predicted_class = CATEGORIES[np.argmax(prediction)]
            print("True Label:", CATEGORIES[selected_label])
            print("Predicted Label:", predicted_class)
            if predicted_class in NUTRIENTS:
                nutrient_info = NUTRIENTS[predicted_class]
                print("Nutrient Information:")
                for nutrient, value in nutrient_info.items():
                    print(f"{nutrient.capitalize()}: {value}g")
            else:
                print("Nutrient information not available.")
        else:
            print("Invalid index. Please try again.")


def classify_image():
    root = Tk()
    root.filename = filedialog.askopenfilename(initialdir="/", title="Select file",
                                                filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
    image_path = root.filename
    root.destroy()
    if image_path:
        try:
            image = Image.open(image_path).convert('RGB')
            image = image.resize(TARGET_SIZE)
            image_np = np.array(image)
            image_np_preprocessed = image_np / 255.0
            image_np_preprocessed = np.expand_dims(image_np_preprocessed, axis=0)
            prediction = model.predict(image_np_preprocessed)
            predicted_class = CATEGORIES[np.argmax(prediction)]
            nutrients = nutrients_and_calories[predicted_class]
            calorie_info = f"Calories: {nutrients['calories']}\nProtein: {nutrients['protein']}g\nCarbs: {nutrients['carbs']}g\nFat: {nutrients['fat']}g"
            cv2.imshow("Classified Image", cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
            print(f"Predicted Class: {predicted_class}")
            print(f"Nutrient and Calorie Information:\n{calorie_info}")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print("Error classifying the image.")
            print(e)


root = Tk()
root.title("Food Image Classifier")

label=Label(root, text="Click the button to classify an image.")
label.pack()
classify_button = Button(root, text="Classify Image", command=classify_image)
classify_button.pack()
root.mainloop()

# Functions definitions end


# main code starts here
print("Following are the Food Categories!")
for file_name in os.listdir(FOOD_CATEGORIES_PATH):
    image_path = os.path.join(FOOD_CATEGORIES_PATH, file_name)
    display(Image.open(image_path))
print("All images are successfully loaded.")


# # Load the arrays from files
# images_data = np.load('images_data.npy')
# labels = np.load('labels.npy')
images_arr = []
labels_arr = []
for category_id, category in enumerate(CATEGORIES):
    for subdirectory in SUBDIRECTORIES:
        subdirectory_path = os.path.join(DATASET_PATH, subdirectory, category)
        images = os.listdir(subdirectory_path)
        for image_name in images:
            image_path = os.path.join(subdirectory_path, image_name)
            try:
                image = Image.open(image_path).convert('RGB')
                image = image.resize(TARGET_SIZE)
                images_arr.append(np.array(image))
            except Exception as e:
                print(f"Error: Unable to read image {image_path}")
                print(e)
        labels_arr.extend([category_id] * len(images))

images_data = np.array(images_arr)
labels = np.array(labels_arr)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(images_data, labels, test_size=0.2, random_state=42)
# X_train_preprocessed = X_train / 255.0
X_test_preprocessed = X_test / 255.0

# Output the sizes
print(f'Total number of Training Samples: {len(X_train)}')
print(f'Total Number of Testing Samples: {len(X_test)}')
print(f'Total number of Samples in Dataset : {len(X_train) + len(X_test)}')

model = load_model('model/food_classifier.keras')

# Evaluate the model on the test data
scores = model.evaluate(X_test_preprocessed, y_test, verbose=0)
print("Test Loss:", scores[0])
print("Test Accuracy:", scores[1])


predictions = model.predict(X_test_preprocessed)
y_pred = np.argmax(predictions, axis=1)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=CATEGORIES, yticklabels=CATEGORIES)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

test_classification(X_test_preprocessed, y_test)
