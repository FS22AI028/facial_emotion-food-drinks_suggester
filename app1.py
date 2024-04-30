# import streamlit as st
# import cv2
# from keras.models import load_model
# from keras.preprocessing.image import img_to_array
# import numpy as np
# import io
# from PIL import Image
# import time

# # Load pre-trained emotion detection model
# classifier = load_model('model.h5')
# emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# # Load face classifier
# face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# def detect_emotion(frame, face_cascade):
#     # Convert the frame to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     # Detect faces in the grayscale frame
#     faces = face_cascade.detectMultiScale(gray)

#     for (x, y, w, h) in faces:
#         # Extract the region of interest (ROI) containing the face
#         roi_gray = gray[y:y+h, x:x+w]
#         # Resize the ROI to fit the input size of the emotion detection model
#         roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
#         # Normalize the pixel values to be in the range [0, 1]
#         roi = roi_gray.astype('float') / 255.0
#         # Convert the ROI to a 3D tensor (height, width, channels) with channels=1
#         roi = img_to_array(roi)
#         roi = np.expand_dims(roi, axis=0)

#         # Perform emotion prediction
#         prediction = classifier.predict(roi)[0]
#         # Get the predicted emotion label
#         label = emotion_labels[prediction.argmax()]
        
#         # Draw a rectangle around the detected face
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
#         # Display the predicted emotion label above the rectangle
#         cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

#     return frame, label

# # Streamlit app code
# def main():
#     st.title("Real-time Facial Emotion Detection")
#     start_button = st.button("Start Capture")
#     stop_button = st.button("Stop Capture")
#     video_placeholder = st.empty()
#     detected_image_placeholder = st.empty()
    
#     # Open a video capture object
#     cap = cv2.VideoCapture(0)
#     start_time = None

#     # Check if the start button is clicked
#     if start_button:
#         st.write("Look at the camera...")
#         while not stop_button:
#             # Read a frame from the video capture
#             ret, frame = cap.read()
#             if not ret:
#                 st.error("Failed to capture video.")
#                 break
            
#             # Perform emotion detection on the frame
#             frame, label = detect_emotion(frame, face_classifier)
            
#             # Display the annotated frame in Streamlit
#             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             pil_img = Image.fromarray(rgb_frame)
#             video_placeholder.image(pil_img, channels="RGB", use_column_width=True)
            
#             # Check if an emotion is detected
#             if label:
#                 # If the timer is not set, set it
#                 if start_time is None:
#                     start_time = time.time()
#                 # If the emotion has been shown for at least 2 seconds
#                 elif time.time() - start_time >= 2:
#                     # Convert the frame to RGB and display below the video
#                     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                     pil_img = Image.fromarray(rgb_frame)
#                     detected_image_placeholder.image(pil_img, channels="RGB", use_column_width=True)
#                     # Stop capturing after displaying the detected emotion for 2 seconds
#                     break
#             else:
#                 start_time = None
    
#     # Release the video capture object
#     cap.release()

# if __name__ == "__main__":
#     main()



# import streamlit as st
# import cv2
# from keras.models import load_model
# from keras.preprocessing.image import img_to_array
# import numpy as np
# import io
# from PIL import Image
# import time

# # Load pre-trained emotion detection model
# classifier = load_model('model.h5')
# emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# # Load face classifier
# face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Define food and drink suggestions based on emotions
# food_and_drink_suggestions = {
#     'Angry': ['Spicy chicken wings', 'Dark chocolate', 'Green tea'],
#     'Disgust': ['Fresh fruits', 'Vegetable soup', 'Herbal tea'],
#     'Fear': ['Comforting soup', 'Warm tea', 'Banana'],
#     'Happy': ['Pizza', 'Ice cream', 'Smoothie'],
#     'Neutral': ['Sandwich', 'Mixed nuts', 'Water'],
#     'Sad': ['Chocolate chip cookies', 'Macaroni and cheese', 'Hot cocoa'],
#     'Surprise': ['Sushi', 'Fruit juice', 'Energy drink']
# }

# def detect_emotion(frame, face_cascade):
#     # Initialize label to None
#     label = None
    
#     # Convert the frame to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     # Detect faces in the grayscale frame
#     faces = face_cascade.detectMultiScale(gray)

#     for (x, y, w, h) in faces:
#         # Extract the region of interest (ROI) containing the face
#         roi_gray = gray[y:y+h, x:x+w]
#         # Resize the ROI to fit the input size of the emotion detection model
#         roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
#         # Normalize the pixel values to be in the range [0, 1]
#         roi = roi_gray.astype('float') / 255.0
#         # Convert the ROI to a 3D tensor (height, width, channels) with channels=1
#         roi = img_to_array(roi)
#         roi = np.expand_dims(roi, axis=0)

#         # Perform emotion prediction
#         prediction = classifier.predict(roi)[0]
#         # Get the predicted emotion label
#         label = emotion_labels[prediction.argmax()]
        
#         # Draw a rectangle around the detected face
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
#         # Display the predicted emotion label above the rectangle
#         cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

#     return frame, label

# # Streamlit app code
# def main():
#     st.title("Real-time Facial Emotion Detection")
#     start_button = st.button("Start Capture")
#     stop_button = st.button("Stop Capture")
#     video_placeholder = st.empty()
#     detected_image_placeholder = st.empty()
#     suggestions_placeholder = st.empty()
    
#     # Open a video capture object
#     cap = cv2.VideoCapture(0)
#     start_time = None

#     # Check if the start button is clicked
#     if start_button:
#         st.write("Look at the camera...")
#         while not stop_button:
#             # Read a frame from the video capture
#             ret, frame = cap.read()
#             if not ret:
#                 st.error("Failed to capture video.")
#                 break
            
#             # Perform emotion detection on the frame
#             frame, label = detect_emotion(frame, face_classifier)
            
#             # Display the annotated frame in Streamlit
#             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             pil_img = Image.fromarray(rgb_frame)
#             video_placeholder.image(pil_img, channels="RGB", use_column_width=True)
            
#             # Check if an emotion is detected
#             if label:
#                 # If the timer is not set, set it
#                 if start_time is None:
#                     start_time = time.time()
#                 # If the emotion has been shown for at least 2 seconds
#                 elif time.time() - start_time >= 2:
#                     # Display the detected emotion for 2 seconds
#                     st.write("Detected Emotion:", label)
#                     # Display food and drink suggestions based on the detected emotion
#                     if label in food_and_drink_suggestions:
#                         suggestions_placeholder.write("Food and Drink Suggestions:")
#                         for suggestion in food_and_drink_suggestions[label]:
#                             suggestions_placeholder.write("- " + suggestion)
#                     # Convert the frame to RGB and display below the video
#                     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                     pil_img = Image.fromarray(rgb_frame)
#                     detected_image_placeholder.image(pil_img, channels="RGB", use_column_width=True)
#                     # Stop capturing after displaying the detected emotion for 2 seconds
#                     break
#             else:
#                 start_time = None
    
#     # Release the video capture object
#     cap.release()

# if __name__ == "__main__":
#     main()





import streamlit as st
import cv2
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import time

# Load pre-trained emotion detection model
classifier = load_model('model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load face classifier
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define food and drink suggestions based on emotions
food_and_drink_suggestions = {
    'Angry': ['Spicy chicken wings', 'Dark chocolate', 'Green tea'],
    'Disgust': ['Fresh fruits', 'Vegetable soup', 'Herbal tea'],
    'Fear': ['Comforting soup', 'Warm tea', 'Banana'],
    'Happy': ['Pizza', 'Ice cream', 'Smoothie'],
    'Neutral': ['Sandwich', 'Mixed nuts', 'Water'],
    'Sad': ['Chocolate chip cookies', 'Macaroni and cheese', 'Hot cocoa'],
    'Surprise': ['Sushi', 'Fruit juice', 'Energy drink']
}

def detect_emotion(frame, face_cascade):
    label = None
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        roi = roi_gray.astype('float') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        prediction = classifier.predict(roi)[0]
        label = emotion_labels[prediction.argmax()]
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return frame, label

def main():
    st.title("Real-time Facial Emotion Detection & Food/Drinks suggester")
    start_button = st.button("Start Capture", key="start")
    stop_button = st.button("Stop Capture", key="stop")
    video_placeholder = st.empty()
    detected_image_placeholder = st.empty()
    suggestions_placeholder = st.empty()

    cap = cv2.VideoCapture(0)
    start_time = None

    if start_button:
        # st.write("Look at the camera...")
        while not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture video.")
                break
            
            frame, label = detect_emotion(frame, face_classifier)
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            video_placeholder.image(pil_img, channels="RGB", use_column_width=True)
            
            if label:
                if start_time is None:
                    start_time = time.time()
                elif time.time() - start_time >= 2:
                    # Display the detected emotion
                    st.write("Detected Emotion:")
                    st.success(f"  {label}")  # You can use st.info, st.warning, or st.error for different styles
                    
                    # Display the suggested food and drink
                    if label in food_and_drink_suggestions:
                        suggestions_placeholder.write("Suggested Food:")
                        for suggestion in food_and_drink_suggestions[label]:
                            suggestions_placeholder.success(f"  {suggestion}")  # You can use st.info, st.warning, or st.error for different styles
                    
                    detected_image_placeholder.image(pil_img, channels="RGB", use_column_width=True)
                    break
            else:
                start_time = None
    
    cap.release()

if __name__ == "__main__":
    main()


