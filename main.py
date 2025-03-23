import os
from fastapi import FastAPI, File, UploadFile
import numpy as np
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
import os


app = FastAPI()

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Load the trained model
#MODEL_PATH = os.path.join(os.path.dirname(__file__), "trained_plant_disease_model.keras")
#MODEL_PATH="trained_plant_disease_model.keras"
#model = tf.keras.models.load_model(MODEL_PATH)

#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#MODEL_PATH = os.path.join(BASE_DIR, "trained_plant_disease_model.keras")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.h5")

print(f"Checking if model exists: {os.path.exists(MODEL_PATH)}")
print(f"Loading model from: {MODEL_PATH}")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)


print(f"Loading model from: {MODEL_PATH}")  # Debugging

model = tf.keras.models.load_model(MODEL_PATH)

# Define class labels
CLASS_NAMES = [
    'Apple_scab', 'Black_rot', 'Cedar_apple_rust', 'healthy', 
    'healthy', 'Powdery_mildew', 'healthy', 'Cercospora_leaf_spot Gray_leaf_spot', 
    'Common_rust_', 'Northern_Leaf_Blight', 'healthy', 'Black_rot', 
    'Esca_(Black_Measles)', 'Leaf_blight_(Isariopsis_Leaf_Spot)', 'healthy', 
    'Haunglongbing_(Citrus_greening)', 'Bacterial_spot', 'healthy', 
    'Bacterial_spot', 'healthy', 'Early_blight', 'Late_blight', 'healthy', 
    'healthy', 'healthy', 'Powdery_mildew', 'Leaf_scorch', 'healthy', 
    'Bacterial_spot', 'Early_blight', 'Late_blight', 'Leaf_Mold', 
    'Septoria_leaf_spot', 'Spider_mites Two-spotted_spider_mite', 
    'Target_Spot', 'Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_mosaic_virus', 'healthy'
]


def preprocess_image(img):
    """Resize and normalize the image for model prediction."""
    img = img.convert("RGB")  # Convert grayscale or RGBA images to RGB
    img = img.resize((128, 128))  # Match model input size
    img = image.img_to_array(img) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Expand dimensions for model
    return img

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Accepts an image, processes it, and returns a prediction."""
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        img = preprocess_image(img)

        prediction = model.predict(img)
        predicted_class = CLASS_NAMES[np.argmax(prediction)]
        confidence = float(np.max(prediction))

        print(f"Prediction probabilities: {prediction}")
        print(f"Predicted class: {predicted_class} with confidence: {confidence}")

        return {"disease": predicted_class, "confidence": confidence}

    except Exception as e:
        return {"error": str(e)}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
  import uvicorn
  port = int(os.environ.get("PORT", 8000))  # Default to 8000 if PORT is not set
  uvicorn.run(app, host="0.0.0.0", port=port)

