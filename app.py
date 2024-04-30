import streamlit as st
import cv2
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np

# Load pre-trained emotion detection model
classifier = load_model('model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load face classifier
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_emotion(frame, face_cascade):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) containing the face
        roi_gray = gray[y:y+h, x:x+w]
        # Resize the ROI to fit the input size of the emotion detection model
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        # Normalize the pixel values to be in the range [0, 1]
        roi = roi_gray.astype('float') / 255.0
        # Convert the ROI to a 3D tensor (height, width, channels) with channels=1
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # Perform emotion prediction
        prediction = classifier.predict(roi)[0]
        # Get the predicted emotion label
        label = emotion_labels[prediction.argmax()]
        
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # Display the predicted emotion label above the rectangle
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    return frame

# Streamlit app code
def main():
    st.title("Real-time Facial Emotion Detection")
    start_button = st.button("Start Capture")
    
    # Open a video capture object
    cap = cv2.VideoCapture(0)

    # Check if the start button is clicked
    if start_button:
        # Read a single frame from the video capture
        ret, frame = cap.read()
        if ret:
            # Perform emotion detection on the frame
            frame = detect_emotion(frame, face_classifier)
            # Display the annotated frame in Streamlit
            st.image(frame, channels="BGR", use_column_width=True)
        else:
            st.error("Failed to capture video.")
    
    # Release the video capture object
    cap.release()

if __name__ == "__main__":
    main()
