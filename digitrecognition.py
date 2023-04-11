import cv2
import numpy as np
import pickle

# Load the MNIST dataset saved as a pickle file
with open('mnist_dataset.pkl', 'rb') as f:
    mnist_dataset = pickle.load(f)

# Extract the training and testing images and labels
train_images = mnist_dataset['train_images']
train_labels = mnist_dataset['train_labels']
test_images = mnist_dataset['test_images']
test_labels = mnist_dataset['test_labels']

# Define a function to preprocess the image
def preprocess_image(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to the image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find the contour of the digit in the image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get the bounding box of the digit
    x, y, w, h = cv2.boundingRect(contours[0])
    
    # Extract the digit from the image
    digit = thresh[y:y+h, x:x+w]
    
    # Resize the digit to 28x28 pixels
    resized_digit = cv2.resize(digit, (28, 28))
    
    # Reshape the digit to a 1D array
    flattened_digit = resized_digit.reshape(1, 784)
    
    # Normalize the pixel values to between 0 and 1
    normalized_digit = flattened_digit / 255.0
    
    return normalized_digit


# Load the model saved as a pickle file
with open('mnist_model.pkl', 'rb') as f:
    model = pickle.load(f)


# Open the camera
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    
    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Resize the frame to 480x480 pixels
    resized = cv2.resize(gray, (480, 480))
    
    # Extract the ROI for digit recognition
    roi = resized[120:360, 120:360]
    
    # Preprocess the ROI
    preprocessed_roi = preprocess_image(roi)
    
    # Reshape the preprocessed ROI to a 3D array
    preprocessed_roi = preprocessed_roi.reshape(1, 28, 28)
    
    # Predict the digit using the model
    prediction = model.predict(preprocessed_roi)
    
    # Convert the prediction to an integer
    digit = int(prediction[0])
    
    # Put the predicted digit on the frame
    cv2.putText(frame, str(digit), (200, 440), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 2)
    
    # Show the frame
    cv2.imshow('frame', frame)
    
    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
