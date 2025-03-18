

import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

import json

# Replace with your actual notebook file path
notebook_path = "test_plant_disease.ipynb"
output_path = "test_plant_disease.py"

# Load the notebook
with open(notebook_path, "r", encoding="utf-8") as f:
    notebook = json.load(f)

# Extract code cells
code_cells = [cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "code"]

# Write to a Python file
with open(output_path, "w", encoding="utf-8") as f:
    for code in code_cells:
        f.write("".join(code) + "\n\n")

print(f"Extracted code saved to {output_path}")


validation_set = tf.keras.utils.image_dataset_from_directory(
    r"C:\Users\sanja\OneDrive\Desktop\Final Year Projecct\Plant_Disease_Prediction\Dataset\Plant_Disease_Dataset\valid",
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)
class_name = validation_set.class_names
print(class_name)

import os
import tensorflow as tf

# Load dataset
test_set = tf.keras.utils.image_dataset_from_directory(
    r"C:\Users\sanja\OneDrive\Desktop\Final Year Projecct\Plant_Disease_Prediction\Dataset\Plant_Disease_Dataset\valid",
    labels="inferred",
    label_mode="categorical",
    image_size=(128, 128),
    batch_size=1,
    shuffle=False
)

# Extract original class names
original_class_names = test_set.class_names

# Function to remove plant names
def clean_class_name(class_name):
    return "___".join(class_name.split("___")[1:])  # Removes the first part (plant name)

# Apply function to get cleaned class names
cleaned_class_names = [clean_class_name(name) for name in original_class_names]

print("Original Class Names:", original_class_names)
print("Cleaned Class Names:", cleaned_class_names)


cnn = tf.keras.models.load_model('trained_plant_disease_model.keras')

import cv2
import matplotlib.pyplot as plt
import os

image_path = r"C:\Users\sanja\OneDrive\Desktop\Final Year Projecct\Plant_Disease_Prediction\Dataset\Plant_Disease_Dataset\test\test\AppleCedarRust1.JPG"

# Check if file exists before reading
if not os.path.exists(image_path):
    print("Error: Image file not found at the specified path.")
else:
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Unable to load image.")
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        # Display the image
        plt.imshow(img)
        plt.title('Test Image')
        plt.xticks([])
        plt.yticks([])
        plt.show()


image = tf.keras.preprocessing.image.load_img(image_path,target_size=(128,128))
input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr])  # Convert single image to a batch.
predictions = cnn.predict(input_arr)

print(predictions)

result_index = np.argmax(predictions) #Return index of max element
print(result_index)



# Displaying the disease prediction
model_prediction = class_name[result_index]
plt.imshow(img)
plt.title(f"Disease Name: {model_prediction}")
plt.xticks([])
plt.yticks([])
plt.show()



