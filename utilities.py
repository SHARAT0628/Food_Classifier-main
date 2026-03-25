import csv
import os
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display
import numpy as np
import pandas as pd

dataset_path = 'food11'
categories = ['apple_pie', 'cheesecake', 'chicken_curry', 'french_fries', 'fried_rice',
              'hamburger', 'hot_dog', 'ice_cream', 'omelette', 'pizza', 'sushi']

csv_file = "image_labels.csv"
DATASET_PATH = 'food11'
CATEGORIES = ['apple_pie', 'cheesecake', 'chicken_curry', 'french_fries', 'fried_rice',
              'hamburger', 'hot_dog', 'ice_cream', 'omelette', 'pizza', 'sushi']
SUBDIRECTORIES = ['train', 'test']
TARGET_SIZE = (224, 224)


# # Open a CSV file to write the image paths and labels
# with open(csv_file, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['image_path', 'label'])  # header row
#
#     for category_id, category in enumerate(categories):
#         for subdirectory in SUBDIRECTORIES:
#             subdirectory_path = os.path.join(dataset_path, subdirectory, category)
#             images = os.listdir(subdirectory_path)
#             for image_name in images:
#                 image_path = os.path.join(subdirectory_path, image_name)
#                 writer.writerow([image_path, category_id])  # write image path and label
#
# print("CSV file has been created with image paths and labels.")

# SAVE_PATH = 'Food Categories'
# os.makedirs(SAVE_PATH, exist_ok=True)
# already_saved = os.listdir(SAVE_PATH)
#
# if not already_saved:  # If no images are saved, process and save them
#     print("Processing and saving images...")
#     for category in CATEGORIES:
#             for subdirectory in SUBDIRECTORIES:
#                 subdirectory_path = os.path.join(DATASET_PATH, subdirectory, category)
#                 images = os.listdir(subdirectory_path)
#                 for image_name in images:
#                     image_path = os.path.join(subdirectory_path, image_name)
#                     try:
#                         if category not in already_saved:
#                             image = Image.open(image_path)
#                             image = image.resize(TARGET_SIZE)
#                             new_image = Image.new("RGB", (224, 224 + 20), "white")
#                             new_image.paste(image, (0, 0))
#                             draw = ImageDraw.Draw(new_image)
#                             font = ImageFont.load_default()
#                             draw.text((112, 224 + 5), category, fill="black", font=font, anchor="mm")
#                             # Save the processed image
#                             save_path = os.path.join(SAVE_PATH, f"{category}.png")
#                             new_image.save(save_path)
#                             print(f"Saved {save_path}")
#                             break  # Save only one image per category
#                     except Exception as e:
#                         print(f"Error: Unable to read image {image_path}")
#                         print(e)
#
#
# data = pd.read_csv('image_labels.csv')
#
# # Prepare the image and label arrays
# images_arr = []
# labels_arr = []
#
# for _, row in data.iterrows():
#     try:
#         image = Image.open(row['image_path']).convert('RGB')
#         image = image.resize(TARGET_SIZE)
#         images_arr.append(np.array(image))
#         labels_arr.append(row['label'])
#     except Exception as e:
#         print(f"Error: Unable to read image {row['image_path']}")
#         print(e)
#
# # Convert lists to numpy arrays
# images_data = np.array(images_arr)
# labels = np.array(labels_arr)
#
# # Save the arrays to files
# np.save('images_data.npy', images_data)
# np.save('labels.npy', labels)
