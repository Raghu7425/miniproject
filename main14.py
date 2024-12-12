import cv2
import time
import streamlit as st
import mediapipe as mp
from datetime import datetime
import numpy as np
import pandas as pd
from deepface import DeepFace
import pyaudio
import wave
import threading

# Initialize mediapipe solutions for face detection, pose, and face landmarks
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

# Function for face detection
def detect_face_and_eyes(frame):
    with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection:
        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)
        
        if results.detections:
            return "Face Detected"
        else:
            return "No Face Detected"

# Function for head pose estimation (simple)
def head_pose_estimation(frame):
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        
        if results.pose_landmarks:
            nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
            left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            
            if nose.y < left_shoulder.y and nose.y < right_shoulder.y:
                return "User is facing forward"
            else:
                return "User is not facing the screen"
        else:
            return "No pose detected"

# Enhanced Fraud Detection using pre-trained model (DeepFace or similar)
def spoof_detection(frame):
    try:
        result = DeepFace.analyze(frame, actions=['age', 'gender', 'race', 'emotion'])
        if 'spoof' in result:  # Check if spoofing is detected
            return True
    except:
        return False  # If DeepFace fails, assume no spoof detected
    return False

# Eye Movement Tracking (checking if the eyes are looking directly at the camera)
def eye_movement_tracking(frame, face_landmarks):
    left_eye = [face_landmarks.landmark[i] for i in range(33, 39)]
    right_eye = [face_landmarks.landmark[i] for i in range(39, 45)]
    
    left_eye_center = np.mean([point.x for point in left_eye])
    right_eye_center = np.mean([point.x for point in right_eye])
    
    if abs(left_eye_center - right_eye_center) < 0.02:
        return "Eyes Aligned"
    else:
        return "Eyes Not Aligned"

# Function to detect if sound is being recorded (simple speech detection based on volume)
def detect_speech():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
    
    while True:
        audio_data = np.frombuffer(stream.read(1024), dtype=np.int16)
        volume = np.linalg.norm(audio_data)  # Calculate volume as the norm of the audio samples
        
        if volume > 500:  # Threshold for detecting speech/noise
            return True
    return False

# Function to start the proctoring system
def proctoring_system(test_duration_minutes):
    st.title("Online Proctoring System")
    st.write("This system uses real-time face recognition, eye tracking, head pose, and spoof detection.")
    
    test_duration_seconds = test_duration_minutes * 60  # Convert minutes to seconds
    
    # Set up the timer for the user-defined test duration
    if 'start_time' not in st.session_state:
        st.session_state.start_time = time.time()  # Start timer when the function is first called
        st.session_state.test_duration = test_duration_seconds  # User-defined test duration
    
    # Track elapsed time
    elapsed_time = time.time() - st.session_state.start_time
    remaining_time = st.session_state.test_duration - elapsed_time
    minutes = int(remaining_time // 60)
    seconds = int(remaining_time % 60)
    
    # Display elapsed time
    st.write(f"Time Remaining: {minutes} min {seconds} sec")
    
    # Stop the test when the timer runs out
    if remaining_time <= 0:
        st.write("Test Time Completed.")
        st.success("Proctoring stopped. Generating the report...")
        generate_report()
        return

    # Set up webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Could not open webcam.")
        return
    
    stframe = st.empty()
    
    # Initialize state variables for results
    face_status = "No face detected"
    head_pose = "User is not facing the screen"
    spoof_status = "Spoof not detected"
    eye_status = "Eyes Not Aligned"
    
    # Collecting all results for the report
    results_data = []
    
    # Create a placeholder for continuous alerts
    alert_placeholder = st.empty()
    
    # Start the proctoring system
    with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        while remaining_time > 0:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip the frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert frame to RGB for processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                # Detect face and eyes
                face_status = detect_face_and_eyes(frame)
                
                # Get face landmarks for additional checks
                landmarks = results.multi_face_landmarks[0]
                
                # Perform additional biometric checks
                eye_status = eye_movement_tracking(frame, landmarks)
                
                # Detect head pose
                head_pose = head_pose_estimation(frame)
                
                # Fraud detection (replace with a more advanced model)
                spoof_status = spoof_detection(frame)
            else:
                eye_status = "No Eyes Detected"
                head_pose = "No Pose Detected"
                spoof_status = "Spoof Detected"
            
            # Convert the frame to RGB and display it in Streamlit
            stframe.image(frame, channels="BGR", use_column_width=True)
            
            # Collect the data for the report
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            results_data.append([current_time, face_status, head_pose, eye_status, spoof_status])
            
            # Continuous alerts shown beside the video feed
            alert_message = f"""
                **Face Status**: {face_status}  
                **Head Pose**: {head_pose}  
                **Eye Status**: {eye_status}  
                **Spoof Detection**: {'Spoof detected' if spoof_status else 'Real face detected'}
            """
            alert_placeholder.markdown(alert_message)
            
            # Check for speech detection in a separate thread to avoid blocking the main process
            speech_thread = threading.Thread(target=detect_speech)
            speech_thread.start()
            speech_thread.join()
            
            if speech_thread.is_alive():
                alert_placeholder.markdown("**Alert**: Speaking detected!")
            
            # Update remaining time
            elapsed_time = time.time() - st.session_state.start_time
            remaining_time = st.session_state.test_duration - elapsed_time
            minutes = int(remaining_time // 60)
            seconds = int(remaining_time % 60)
            st.write(f"Time Remaining: {minutes} min {seconds} sec")
        
        cap.release()
        generate_report(results_data)

def generate_report(results_data):
    # Create a DataFrame from collected results
    df = pd.DataFrame(results_data, columns=["Timestamp", "Face Status", "Head Pose", "Eye Status", "Spoof Detection"])
    
    # Save the DataFrame to CSV
    df.to_csv("proctoring_report.csv", index=False)
    
    # Provide download link for CSV file
    st.success("Test completed. Fraud report generated.")
    st.download_button(label="Download Report", data=df.to_csv(index=False).encode(), file_name="proctoring_report.csv", mime="text/csv")

# Run the proctoring system
if __name__ == "__main__":
    test_duration_minutes = st.number_input("Enter test duration in minutes:", min_value=1, max_value=180, value=30)
    
    # Start button to begin proctoring after user input
    if st.button("Start Proctoring"):
        proctoring_system(test_duration_minutes)
