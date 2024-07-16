import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json

# Set the environment variable for encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Load the model architecture
try:
    with open('./final_model/garbage_classification_inception.json', 'r', encoding='utf-8') as json_file:
        model_json = json_file.read()
    print("Model architecture loaded successfully.")
except UnicodeEncodeError as e:
    print(f"Error reading model architecture: {e}")
    exit(1)

model = model_from_json(model_json)

# Load the model weights
try:
    model.load_weights('./final_model/garbage_classification_inception_weights.h5')
    print("Model weights loaded successfully.")
except Exception as e:
    print(f"Error loading model weights: {e}")
    exit(1)

# Define the class names (replace these with your actual class names)
class_names = ["battery", "biological", "clothes", "glass", "metal", "paper", "plastic", "shoes", "trash"]

# Function to preprocess the frame
def preprocess_frame(frame):
    img = cv2.resize(frame, (400, 400))  # Resize the frame to the input size of the model
    img = img.astype('float32') / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Expand dimensions to fit the model input
    return img

# Function to get the class name from prediction
def get_class_name(prediction):
    class_idx = np.argmax(prediction)
    return class_names[class_idx]

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # Capture frame-by-frame
    if not ret:
        break
    
    # Preprocess the frame
    preprocessed_frame = preprocess_frame(frame)
    
    # Make prediction
    try:
        prediction = model.predict(preprocessed_frame)
        class_name = get_class_name(prediction)
        confidence = np.max(prediction)
    except Exception as e:
        print(f"Error during prediction: {e}")
        continue
    
    # Draw bounding box and label
    label = f"{class_name}: {confidence:.2f}"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.rectangle(frame, (10, 10), (frame.shape[1]-10, frame.shape[0]-10), (0, 255, 0), 2)  # Draw a rectangle around the object
    
    # Display the resulting frame
    cv2.imshow('Object Detection', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
