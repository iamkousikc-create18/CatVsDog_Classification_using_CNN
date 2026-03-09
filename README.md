🐶🐱 Dogs vs Cats Image Classification using CNN & Streamlit

📌 Project Overview
This project implements a Convolutional Neural Network (CNN) to classify images as Dog 🐶 or Cat 🐱.
The model is trained on 25,000 labeled images and deployed as a web application using Streamlit, allowing users to upload an image and receive real-time predictions.
Final Validation Accuracy: ~84%

📥 Dataset
Dataset Source (Microsoft Official):
https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip�
After extraction, folder structure should be:

PetImages/
 ├── Cat/
 └── Dog/
Total Images: 25,000
Labels:
0 → Cat
1 → Dog
Corrupted images were removed during preprocessing.

🧠 Model Architecture
The CNN model consists of:
Conv2D (16 filters) + MaxPooling
Conv2D (32 filters) + MaxPooling
Conv2D (64 filters) + MaxPooling
Flatten Layer
Dense Layer (512 neurons, ReLU)
Output Layer (1 neuron, Sigmoid)

⚙️ Training Details
Optimizer: Adam
Loss Function: Binary Crossentropy
Input Shape: (128, 128, 3)
Epochs: 10
Data Augmentation:
Rescaling
Rotation
Zoom
Horizontal Flip

🚀 Streamlit Deployment
The trained model (dogs_vs_cats_model.h5) is deployed using Streamlit.
Features:
Upload JPG / PNG / JPEG image
Automatic resizing (128x128)
Image normalization
Real-time prediction
Simple and clean UI

🛠️ Installation & Setup
1️⃣ Create Environment

conda create -n py11_env python=3.11
conda activate py11_env
2️⃣ Install Dependencies

pip install tensorflow streamlit numpy pandas matplotlib scikit-learn pillow
3️⃣ Run the Application

python -m streamlit run app.py
📁 Project Structure
Copy code

├── app.py
├── dogs_vs_cats_model.h5
├── Dogs vs Cats Image Classification - CNN.ipynb
├── PetImages/
├── requirements.txt
└── README.md

🎯 Future Improvements
Improve accuracy using Transfer Learning (VGG16 / ResNet)
Add confidence score display
Deploy on Streamlit Cloud
Optimize model performance

👨‍💻 Author
Kousik Chakraborty
