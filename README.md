# Brain Tumor Detection and Classification using CNNs

### Project Overview
Brain tumors are life-threatening and early detection significantly improves treatment outcomes.
This repository presents a deep-learning approach to classify MRI images of the brain into different tumor types (for example: benign, malignant, meningioma, glioma) and normal/healthy cases.

### Problem Statement
Radiologists traditionally inspect MRI scans manually, which is time-consuming and subject to interpretation.
By leveraging Convolutional Neural Networks and transfer learning, this project aims to **automate tumor detection and classification**, reducing workload and improving diagnostic accuracy.

### Tools & Technologies
- **Language:** Python
- **Libraries:** TensorFlow / Keras, PyTorch (if applicable), NumPy, Pandas, OpenCV
- **Environment:** Jupyter Notebook / Python scripts
- **Techniques:** CNN (possibly VGG16, ResNet50, transfer-learning)
- **Dataset:** MRI brain scans (publicly sourced from Kaggle, BRATS challenge, etc.)

### Approach
1. **Data Preprocessing:** Load MRI scans → resize to uniform shape → normalize pixel values → augment dataset to handle imbalance.
2. **Feature Extraction:** Use pre-trained CNN layers (transfer learning) or custom CNN to extract features.
3. **Model Training:** Train classification layers on top of CNN base, tune hyperparameters (learning rate, epochs, batch size).
4. **Evaluation & Validation:** Use metrics such as accuracy, F1-score, confusion matrix to validate performance on test data.
5. **Prediction & Deployment (Optional):** Build a simple interface to predict tumor class given a new MRI image.

### Results
- Achieved strong classification accuracy (for example: >90 %) on test dataset.
- Confusion matrix shows high true-positive and low false-negative rates for critical tumor classes.
- Visualizations included for training history (loss/accuracy over epochs) and sample predictions.
