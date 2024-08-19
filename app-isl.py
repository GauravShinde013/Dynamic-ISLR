import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
# MediaPipe drawing library
mp_drawing = mp.solutions.drawing_utils

# Load the trained model
model = tf.keras.models.load_model('D:\MSc AI\Thesis\Streamlit\sign_language_model.keras')

# Sample class labels for testing
class_labels = ['love', 'are you free today', 'can you help me', 'you need a medicine, take this one', 'what is your phone number',
                'could you please talk slower', 'pour some more water into the glass', 'what do you think']

# Initialize MediaPipe Holistic model 
mp_holistic = mp.solutions.holistic.Holistic(
    static_image_mode=False, 
    model_complexity=1, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5)

def extract_keypoints(frame, holistic):
    """
    Extracts the keypoints from a given frame using the holistic model.

    Args:
        frame (numpy.ndarray): The input frame.
        holistic (Holistic): The holistic model.

    Returns:
        tuple: A tuple containing the keypoints and the results.
    """
    results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    keypoints = []
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            keypoints.extend([landmark.x, landmark.y, landmark.z])
    if results.left_hand_landmarks:
        for landmark in results.left_hand_landmarks.landmark:
            keypoints.extend([landmark.x, landmark.y, landmark.z])
    if results.right_hand_landmarks:
        for landmark in results.right_hand_landmarks.landmark:
            keypoints.extend([landmark.x, landmark.y, landmark.z])
    keypoints.extend([0] * (258 - len(keypoints)))  
    return keypoints, results


# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Extract keypoints from the current frame
    keypoints, results = extract_keypoints(frame, mp_holistic)
    keypoints = np.array(keypoints).reshape(1, 1, -1)  # Reshape for model input

    # Draw the landmarks
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)

    # Make a prediction using the trained model
    prediction = model.predict(keypoints)
    print("Prediction:", prediction)
    predicted_index = np.argmax(prediction)
    predicted_label = class_labels[predicted_index] if predicted_index < len(class_labels) else "Unknown"

    # Display the prediction on the frame
    cv2.putText(frame, f'Prediction: {predicted_label}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Sign Language Prediction', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
mp_holistic.close()
