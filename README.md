# Age-and-Gender-Prediction-
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

age_model = tf.keras.models.load_model('age_model.keras')
gender_model = tf.keras.models.load_model('gender_classification_cnn_model.h5')

st.title("Age and Gender Prediction from Image")
uploaded_image = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    image_rgb = image.convert('RGB')
    image_age = image_rgb.resize((200, 200))
    image_gender = image_rgb.resize((100, 100))

    image_age_array = np.array(image_age)
    image_gender_array = np.array(image_gender)

    image_age_array = image_age_array / 255.0
    image_gender_array = image_gender_array / 255.0

    image_age_array = np.expand_dims(image_age_array, axis=0)
    image_gender_array = np.expand_dims(image_gender_array, axis=0)

    age_prediction = age_model.predict(image_age_array)
    st.success(f"Predicted Age: {age_prediction[0][0]:.2f} years")

    gender_prediction = gender_model.predict(image_gender_array)

    if gender_prediction[0][0] > 0.5:
        st.success("Gender: Male")
    else:
        st.success("Gender: Female")
