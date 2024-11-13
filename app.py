import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# Load the trained model
model = tf.keras.models.load_model("best_xception_model.keras")

# Define Hb threshold for anemia diagnosis
hb_threshold = 12.0  # Example threshold (g/dL)

# Function to preprocess the input image
def preprocess_image(image):
    try:
        # Resize the image and normalize
        img = image.resize((224, 224))
        img = np.array(img) / 255.0  # Normalize the image
        return img
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

# Function to predict the Hb value and provide the diagnosis
def predict_hb_and_diagnose(image):
    # Preprocess the image
    processed_image = preprocess_image(image)
    if processed_image is None:
        return "Error processing image", "Unable to predict"

    # Make a prediction using the model
    input_image = np.expand_dims(processed_image, axis=0)  # Add batch dimension
    predicted_hb_value = model.predict(input_image)[0][0]

    # Diagnose anemia based on predicted Hb value
    def diagnose_anemia(hb_value, threshold=hb_threshold):
        if hb_value < threshold:
            return "Anemia detected"
        else:
            return "No anemia"

    diagnosis = diagnose_anemia(predicted_hb_value)

    # Return the result
    return f"Predicted Hb value: {predicted_hb_value:.2f}", diagnosis

# Streamlit UI components
st.title("Hemoglobin Prediction & Anemia Diagnosis")
st.write("Upload an image of the blood sample, and the model will predict the hemoglobin value and diagnose whether the person has anemia.")

# Upload image
image = st.file_uploader("Upload Image of Blood Sample", type=["jpg", "jpeg", "png"])

if image is not None:
    # Open image
    image = Image.open(image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict Hb value and diagnose anemia
    predicted_hb_value, diagnosis = predict_hb_and_diagnose(image)

    # Display prediction and diagnosis
    st.write(predicted_hb_value)
    st.write(diagnosis)

