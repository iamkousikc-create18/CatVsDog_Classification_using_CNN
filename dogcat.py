import io
from io import BytesIO
from PIL import Image
import numpy as np
import pandas as pd
from tensorflow import keras
import keras
from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn

# --- 1. Load Model (Ensure path is correct) ---
# Ensure this path points to your actual trained model file
model = keras.models.load_model(r'C:\Users\USER\Desktop\MLProject\cd.keras', compile=False)

# --- 2. Initialize FastAPI ---
app = FastAPI()

# --- 3. Root Endpoint ---
@app.get('/')
def index():
    return {'Deployment': 'Hello and Welcome to 5 Minutes Engineering'}

# --- 4. Prediction Endpoint ---
@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    # Define the required input size based on your model's training
    TARGET_SIZE = (100, 100) 
    
    try:
        # Read the uploaded file asynchronously
        image_bytes = await file.read()
        
        # Open the image using PIL
        image = Image.open(BytesIO(image_bytes))
        
        # Convert to RGB (standard)
        image = image.convert('RGB')
        
        # 💡 CRITICAL FIX: Resize the image to match the model's input shape (100x100)
        image = image.resize(TARGET_SIZE)

        # Convert PIL image to a numpy array
        pic = np.array(image)
        
        # Normalize the pixel values (as done in training: pic / 255)
        pic = pic / 255
        
        # Add the batch dimension (from (100, 100, 3) to (1, 100, 100, 3))
        pic = np.expand_dims(pic, axis=0)
        
        # --- Make Prediction ---
        predicted_array = model.predict(pic)
        
        # The model uses a sigmoid activation, giving a probability (a single value between 0 and 1)
        # predicted_array will be an array like [[0.85]]
        probability = predicted_array[0][0]
        

        if probability > 0.5:
            output = 'Dog'
        else:
            output = 'Cat'
        
        # Return the prediction and the probability (optional, but helpful)
        return {
            "prediction": output, 
            "probability": float(probability)
        }

    except Exception as e:
        # Return a 500 status code for internal server errors
        raise HTTPException(
            status_code=500, 
            detail=f"An error occurred during prediction: {e}"
        )

# --- 5. Run Uvicorn Server ---
if __name__ == "_main_":
    uvicorn.run(app, host="127.0.0.1", port=5000)