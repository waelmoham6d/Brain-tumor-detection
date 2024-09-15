import streamlit as st
from keras.models import load_model # type: ignore
from keras.preprocessing import image # type: ignore
import numpy as np
from PIL import Image
import cv2
model_NN = load_model(r'C:\Users\mwael\OneDrive\Desktop\home\course\brain_tumor\brain_tumor.keras')

st.title("Brain Tumor MRI 🧠")

uploaded_file = st.file_uploader("Please upload a brain MRI image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
   
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    if img.mode != 'RGB':
        img = img.convert('RGB')

    def prepare_image(image):
        if image is None:
            raise ValueError("Failed to load image. Please check the path.")
        image_array = np.array(image)

        resized_image = cv2.resize(image_array, (256, 256))

        resized_image = resized_image.astype('float32') / 255.0

        image_array = np.expand_dims(resized_image, axis=0)

        return image_array


    prepared_image = prepare_image(img)

    prediction = model_NN.predict(prepared_image)
    predicted_class = np.argmax(prediction, axis=1)

    if predicted_class==0 :
        st.write("The model predicts: **glioma**")
    elif predicted_class==1 :
        st.write("The model predicts: **meningioma**")
    elif predicted_class==2 :
        st.write("The model predicts: **notumor** ALHAMDULLAH ") 
    else:
        st.write("The model predicts: **pituitary**")