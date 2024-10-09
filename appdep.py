import streamlit as st
from io import BytesIO
from PIL import Image
import pickle
import cv2 as cv
import base64
import numpy as np

def get_base64(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


# def set_background(png_file):
#     bin_str = get_base64(png_file)
#     page_bg_img = f"""
#     <style>
#     .stApp {{
#     background-image: url("data:image/png;base64,{bin_str}");
#     background-size: cover;
#     backdrop-filter: blur(1000px);
#     color: black;
#     }}
#     </style>
#     """
#     st.markdown(page_bg_img, unsafe_allow_html=True)

# Preprocess the image
def preprocess_image(image):
    # Resize image to desired dimensions (256x256)
    resized_image = image.resize((256, 256))
    
    # Convert resized image to NumPy array
    resized_array = np.array(resized_image)
    
    # Flatten image
    flattened_image = resized_array.flatten()
    
    # Reshape image to have 196608 features
    reshaped_image = flattened_image.reshape(1, -1)
    
    return reshaped_image

def preprocess_images_for_log(image1):
    # Resize image to desired dimensions (256x256) and convert to RGB
    resized_image1 = image1.resize((256, 256)).convert("RGB")
    
    # Convert resized image to NumPy array
    resized_array1 = np.array(resized_image1)
    
    # Normalize pixel values to range [0, 1]
    normalized_image1 = resized_array1 / 255.0
    
    return normalized_image1.reshape(1, 256, 256, 3)

# Function to make predictions using the chosen model
def predict_image(image, model_name):
    preprocessed_image = preprocess_image(image)
    preprocessed_image1 = preprocess_images_for_log(image)
    if model_name == "Random Forest":
        prediction = rf_model.predict(preprocessed_image)
        probability = rf_model.predict_proba(preprocessed_image)[0][prediction[0]]
        return prediction[0], probability
    elif model_name == "SVM":
        decision_scores = svm_model.decision_function(preprocessed_image)
        probability = 1 / (1 + np.exp(-decision_scores))
        prediction = (probability >= 0.5).astype(int)
        return prediction[0], probability[0]
    elif model_name == "Logistic Regression":
        raw_output = log_model.predict(preprocessed_image1)[0][0]
        probability = 1 / (1 + np.exp(-raw_output))  # Calculate probability using the sigmoid function
        prediction = int(raw_output >= 0.5)
        return prediction, probability
    else:
        return ("Invalid Image")
    
# Main function
def main():
    #set_background(r'1333915.jpg')
    
    st.markdown('<p style="color:#FFFFFF;font-size:50px;border-radius:2%;"> Intoxication Analysis </p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Project description
    st.markdown('<p style="color:#FFFFFF;font-size:33px;border-radius:2%;">Project Description</p>', unsafe_allow_html=True)
    st.markdown('<p style="color:#FFFFFF;font-size:24px;border-radius:2%;">Welcome to the Intoxication Analysis app! This app allows you to upload an image and predicts whether the person in the image appears to be intoxicated or not. Please upload an image and click the \'Classify\' button to get the prediction.</p>', unsafe_allow_html=True)

    st.markdown("---")

    # Upload image
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg"])

    if uploaded_image is not None:
        # Display uploaded image
        st.subheader('Uploaded Image')
        image = Image.open(uploaded_image)
        st.image(image, caption='', use_column_width=True)
        st.markdown("---")

        model_choice = st.radio("Select Model", ("Random Forest", "SVM", "Logistic Regression"))

        if st.button('Classify'):
            # Make prediction
            prediction = predict_image(image, model_choice)

            # Display prediction
            st.subheader('Prediction')
            #st.write(prediction)
            st.markdown(f'''<p style ="font-size:30px">{prediction}</p>''', unsafe_allow_html = True)

if __name__ == "__main__":
    # Load the pickle files for the models
    with open('C:\\Python Workspace\\Project X\\rfc_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
        
    with open('C:\\Python Workspace\\Project X\\svm_model.pkl', 'rb') as f:
        svm_model = pickle.load(f)
        
    with open('C:\\Python Workspace\\Project X\\logistic_regression_model.pkl', 'rb') as f:
        log_model = pickle.load(f)

    main()
