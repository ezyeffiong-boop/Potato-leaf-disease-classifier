
import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model(
    "/content/drive/MyDrive/GET324_Project/Model/potato_leaf_classifier.keras"
    )

    # Class names
class_names = ["Potato_late_blight", "potato_healthy"]

    # Title
st.title("Potato Leaf Disease Classification")

st.write("Upload a potato leaf image to predict whether it is Healthy or has Late Blight.")

    # Upload image
uploaded_file = st.file_uploader(
        "Choose a potato leaf image",
            type=["jpg", "jpeg", "png"]
            )

if uploaded_file is not None:
  img = Image.open(uploaded_file)
  st.image(img, caption="Uploaded Image", use_container_width=True)

                        # Resize image
  img = img.resize((256, 256))
  img_array = image.img_to_array(img)
  img_array = np.expand_dims(img_array, axis=0)

  # Predictionprediction = model.predict(img_array)
  prediction = model.predict(img_array)
  predicted_class = np.argmax(prediction)
  confidence = np.max(prediction) * 100

  st.success(f"Prediction:{class_names[predicted_class]}")
  st.info(f"Confidence: {confidence:.2f}%")
