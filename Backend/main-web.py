# Brain Tumor Detection API
# Using FastAPI and a pre-trained TensorFlow CNN model

# Import required libraries
from fastapi import FastAPI, File, UploadFile       # FastAPI for API creation and handling file uploads
import uvicorn                                      # Used to run the FastAPI app
import numpy as np                                  # Numerical operations
import cv2                                          # OpenCV for image decoding and resizing
import tensorflow as tf                             # TensorFlow for loading and running the trained CNN model
from fastapi.middleware.cors import CORSMiddleware  # Enables Cross-Origin Resource Sharing (CORS)

# Initialize FastAPI app
app = FastAPI()

# Define allowed origins (frontend URLs that can interact with this API)
origins = [
    'http://localhost',
    'http://localhost:3000',
]

# Configure CORS middleware to allow API access from different domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],          # Allow all origins (can be restricted for production)
    allow_credentials=True,
    allow_methods=['*'],          # Allow all HTTP methods (POST, GET, etc.)
    allow_headers=['*'],          # Allow all headers
)

# Load trained TensorFlow model
MODEL = tf.keras.models.load_model('../saved_models/2')

# Define class labels corresponding to model output indices
CLASS_NAMES = ['No Tumor', 'Glioma Tumor', 'Meningioma Tumor', 'Pituitary Tumor']

# Utility function: Read and decode uploaded image
def read_image(contents) -> np.ndarray:
    """
    Converts uploaded image bytes into an OpenCV-readable NumPy array.
    Args:
        contents (bytes): Raw image data uploaded via API.
    Returns:
        img (numpy.ndarray): Decoded image in BGR format.
    """
    nparr = np.frombuffer(contents, np.uint8)     # Convert bytes to 1D NumPy array
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)   # Decode array into color image (BGR)
    return img

# POST API Endpoint: /predict
# Accepts an uploaded MRI image and returns tumor prediction
@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    """
    Endpoint to predict brain tumor type from uploaded MRI image.
    Accepts an image file and returns predicted tumor class and confidence score.
    """
    # Read and decode uploaded image
    image = read_image(await file.read())

    # Resize image to match model input dimensions (256x256)
    img_resized = cv2.resize(image, dsize=(256, 256))

    # Expand dimensions to create a batch of size 1 (model expects batch input)
    img_batch = np.expand_dims(img_resized, axis=0)

    # Perform prediction using the loaded CNN model
    prediction = MODEL.predict(img_batch)

    # Get predicted class and confidence score
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Return JSON response with class and confidence
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

# Run FastAPI server (development mode)
if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8000)
