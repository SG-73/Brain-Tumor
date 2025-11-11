# Brain Tumor Detection and Classification using CNNs

### Overview
This project provides a **Streamlit-based web application** for detecting and classifying brain tumors from MRI scans using a **Convolutional Neural Network (CNN)** built with TensorFlow/Keras.
Users can upload MRI images to get real-time predictions for tumor types such as **No Tumor, Glioma, Meningioma, and Pituitary Tumor**.

**Dataset:** [Kaggle - Brain Tumors Dataset](https://www.kaggle.com/datasets/mohammadhossein77/brain-tumors-dataset)

### Problem Statement
Manual identification of brain tumors in MRI scans is labor-intensive and prone to human error.
This project leverages **deep learning** to automate tumor detection and classification, improving diagnostic **speed, consistency, and accessibility**.

### Tools & Technologies
- **Language:** Python
- **Frameworks & Libraries:** Streamlit, TensorFlow/Keras, NumPy, Pillow, OpenCV
- **Model:** CNN models trained and compared in the notebook **`Project\Brain Tumor Detection using 4 variants of CNN (Final).ipynb`**; the best-performing model is saved and loaded for inference in the app.
- **Deployment:** Streamlit Cloud
- **Environment:** Python script-based Streamlit Web App

### Approach
**Model Loading:** The best-performing CNN (from the notebook experiments) is loaded at runtime for inference.
**Image Preprocessing:** Uploaded MRI images are resized (256×256) and normalized to match the model’s input.
**Prediction:** The CNN outputs the tumor class and confidence score in real time.
**Visualization:** The uploaded image and prediction results are displayed side-by-side for clarity.
**Information Display:** For tumor cases, brief medical insights about the detected tumor type are shown.

### Results
- Provides **real-time tumor classification** with high accuracy across four categories.
- Delivers **confidence levels** for each prediction.
- Offers an intuitive and visually informative web interface.

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
