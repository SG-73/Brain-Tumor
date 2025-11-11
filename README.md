# Brain Tumor Detection and Classification using CNNs

### DATASET: https://www.kaggle.com/datasets/mohammadhossein77/brain-tumors-dataset

### Project Overview
This project provides a web-based application for detecting brain tumors from MRI images using a Convolutional Neural Network (CNN) built and trained in TensorFlow/Keras.
The Streamlit-based interface allows users to upload an MRI scan and instantly receive a prediction indicating whether the scan shows a tumor — and, if so, the type of tumor.

### Problem Statement
Detecting brain tumors manually from MRI images can be time-consuming and prone to human error.
This project automates the process using a trained deep-learning model, improving the speed, consistency, and accessibility of tumor diagnosis.

### Tools & Technologies
- **Language:** Python
- **Frameworks & Libraries:** Streamlit, TensorFlow/Keras, NumPy, Pillow
- **Environment:** Streamlit Web App (.py script)
- **Model Used:** Pre-trained CNN (loaded via tf.keras.models.load_model)
- **Dataset:** MRI brain scans (publicly sourced from Kaggle)

### Approach
1. **Model Loading:** The trained CNN model is loaded dynamically using TensorFlow/Keras from the Models/2 directory.
2. **Image Preprocessing:** Uploaded MRI images are resized and normalized to ensure compatibility with the model’s input requirements.
3. **Prediction:** The model processes the image and predicts the most probable tumor class along with a confidence score.
4. **Visualization:** The uploaded image and prediction results are displayed side-by-side for better interpretability.
5. **Information Display:** For tumor cases, the app provides brief medical insights about the detected tumor type.

### Results
- Provides real-time predictions with high confidence for 4 brain tumor classes.
- Interactive and user-friendly web interface for visualizing results.
- Displays both prediction confidence and tumor-specific medical insights.

### How to Run Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/SG-73/Brain-Tumor-Detection.git
   cd Brain-Tumor-Detection
   ```

2. Install dependencies:

   pip install -r requirements.txt

3. Run the app:

   streamlit run app.py

4. Access the app locally at http://localhost:8501
