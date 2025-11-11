# Brain Tumor Detection Web App
# Built using Streamlit and a pre-trained TensorFlow CNN model

# Import necessary libraries
import streamlit as st
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

# App Title and UI Header
st.header(":red[Brain Tumor] Detection :brain:", divider="rainbow")

# File uploader widget – allows users to upload a brain MRI image
upload_file = st.file_uploader("Upload an Image")

# Prediction Section
if upload_file is not None:  # Proceed only when an image is uploaded
    if st.button("Predict"):  # Perform prediction when 'Predict' button is clicked
        
        # Create two columns for side-by-side display (image on left, result on right)
        col1, col2 = st.columns(2)

        # Left Column: Model Loading & Prediction

        with col1:
            # Load the pre-trained CNN model for brain tumor classification
            MODEL = tf.keras.models.load_model("Models/2")

            # Define class labels corresponding to model output indices
            CLASS_NAMES = ["No Tumor", "Glioma Tumor", "Meningioma Tumor", "Pituitary Tumor"]

            # Display the uploaded image
            st.image(upload_file, caption="Uploaded Image", width=300)

            # Convert uploaded image into a NumPy array
            img = np.array(Image.open(BytesIO(upload_file.getvalue())))

            # Resize the image to match model input dimensions
            img = tf.image.resize(img, [256, 256])

            # Expand dimensions to create batch of size 1 (required for prediction)
            img_bt = np.expand_dims(img, 0)

            # Make prediction using the trained model
            prediction = MODEL.predict(img_bt)[0]

            # Identify predicted class (the one with highest probability)
            prediction_class = CLASS_NAMES[np.argmax(prediction)]

            # Calculate confidence score (highest probability × 100)
            confidence = round(np.max(prediction) * 100, 2)

        # Right Column: Display Results

        with col2:
            # If model predicts 'No Tumor', show green success message
            if prediction_class == "No Tumor":
                st.title(f":green[{prediction_class}]")
                st.subheader(f"({confidence}% confidence)")

            # Otherwise, show detailed info for tumor type
            else:
                tab1, tab2 = st.tabs(["🧠Result", "📝Info"])

                # Tab 1: Display predicted tumor type and confidence
                with tab1:
                    st.title(f":red[{prediction_class}]")
                    st.subheader(f"({confidence}% confidence)")

                # Tab 2: Show medical information about the predicted tumor
                with tab2:
                    if prediction_class == "Glioma Tumor":
                        st.write(
                            "Glioma is a type of brain tumor that originates in glial cells, "
                            "which support and protect neurons in the brain. These tumors vary in aggressiveness "
                            "and are classified by cell type and location. Symptoms often include headaches, seizures, "
                            "and neurological deficits. Treatment depends on grade and may involve surgery, radiation, "
                            "and chemotherapy. Prognosis varies with tumor type and stage."
                        )

                    elif prediction_class == "Meningioma Tumor":
                        st.write(
                            "Meningioma arises from the meninges — the tissue layers covering the brain and spinal cord. "
                            "These tumors are usually benign and slow-growing. Large meningiomas may cause headaches, seizures, "
                            "or vision changes. Treatments include observation, surgery, or radiation. Prognosis is generally favorable, "
                            "especially for benign tumors that are completely removed."
                        )

                    else:  # Pituitary Tumor
                        st.write(
                            "A pituitary tumor is an abnormal growth in the pituitary gland at the brain's base. "
                            "Most are benign but can cause hormonal imbalances leading to headaches, vision issues, "
                            "and weight or appetite changes. Treatment may include medication, surgery, or radiation. "
                            "Regular monitoring is essential to maintain hormonal balance and prevent complications."
                        )
