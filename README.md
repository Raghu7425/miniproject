This project is an Online Proctoring System that leverages real-time biometric and behavioral analysis techniques for fraud detection during online tests. The system utilizes computer vision and audio processing to track face status, head pose, eye movement, and detect speech to ensure test integrity.
Features
- **Face Detection**: Detects if a candidate's face is visible.
- **Head Pose Estimation**: Determines if the candidate is facing the screen.
- **Spoof Detection**: Uses a pre-trained DeepFace model to detect fraudulent attempts.
- **Eye Movement Tracking**: Monitors if the candidate is looking directly at the camera.
- **Speech Detection**: Alerts for unauthorized speech or background noise.
- **Real-Time Alerts**: Displays continuous updates on biometric and behavioral checks.
- **Report Generation**: Provides a detailed report of the test session in CSV format.
Requirements
Python Libraries
The following Python libraries are required:
- **cv2**: For webcam access and image processing.
- **mediapipe**: For face, pose, and landmark detection.
- **numpy**: For numerical operations.
- **pandas**: For report generation.
- **streamlit**: For user interface.
- **pyaudio**: For audio recording.
- **deepface**: For advanced face analysis.
- **datetime**: For timestamping events.
- **wave**: For audio manipulation.
- **threading**: For parallel processing.
Installation
Install the required libraries using pip:
```bash
pip install -r requirements.txt
```
How to Run
1. **Start the System**:
   - Run the script in your terminal using:
     ```bash
     streamlit run main14.py
     ```
2. **Set Test Duration**:
   - Input the test duration in minutes (minimum: 1, maximum: 180).
3. **Start Proctoring**:
   - Click the **Start Proctoring** button to begin the session.
4. **Real-Time Monitoring**:
   - The system will capture live video and audio to monitor the candidate’s behavior.
5. **Generate Report**:
   - At the end of the session, the system generates a CSV report summarizing all events.
Outputs
- **Video Feed**: Real-time display of the candidate’s webcam feed.
- **Alerts**: Continuous alerts for biometric and behavioral anomalies.
- **CSV Report**: Downloadable file containing detailed event logs.
Notes
- Ensure the webcam and microphone are functional.
- Use a quiet and well-lit environment for accurate detection.
- The system may require initial setup permissions for accessing webcam and microphone.
Known Issues
- **Performance**: Complex tasks like DeepFace analysis may slow down on systems with lower specifications.
- **Accuracy**: False positives may occur under poor lighting or noisy environments.
Contributions
Feel free to contribute to the project by improving detection algorithms or adding new features. Submit pull requests to the repository.
License
This project is open-source and licensed under the MIT License.
---
For queries or issues, contact the developer.
