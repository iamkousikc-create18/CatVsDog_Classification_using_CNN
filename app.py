import streamlit as st
import tensorflow as tf
from keras.utils import load_img, img_to_array
import numpy as np
from PIL import Image

# 1. Load the pre-trained model
@st.cache_resource # Keeps the model in memory
def load_my_model():
    return tf.keras.models.load_model('dogs_vs_cats_model.h5')

model = load_my_model()

st.title("Dogs vs. Cats Classifier")

# 2. Image Upload Component
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # 3. Preprocessing (Mirroring your training logic)
    # Resize to 128x128 as used in your Conv2D input_shape [cite: 184]
    img = image.resize((128, 128)) 
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize [cite: 323]
    img_array = np.expand_dims(img_array, axis=0) # Reshape for prediction [cite: 324]

    # 4. Prediction logic
    if st.button('Classify'):
        prediction = model.predict(img_array)
        # Using your threshold logic [cite: 326]
        if prediction[0] > 0.5:
            st.write("## Result: It's a Dog! 🐶")
        else:
            st.write("## Result: It's a Cat! 🐱")