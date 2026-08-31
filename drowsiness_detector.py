import cv2
import numpy as np
from scipy.spatial import distance
from collections import deque
import mediapipe as mp

class DrowsinessDetector:
    """
    Detects driver drowsiness using eye closure detection and head position analysis
    """
    def __init__(self, eye_aspect_ratio_threshold=0.2, drowsiness_frames_threshold=20):
        self.eye_aspect_ratio_threshold = eye_aspect_ratio_threshold
        self.drowsiness_frames_threshold = drowsiness_frames_threshold
        self.drowsiness_counter = 0
        self.drowsiness_alert = False
        self.eye_closure_history = deque(maxlen=30)
        self.head_position_history = deque(maxlen=30)
        self.blink_count = 0
        self.last_blink_time = 0
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Eye landmarks indices
        self.LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]
        
    def calculate_eye_aspect_ratio(self, eye_landmarks):
        """
        Calculate Eye Aspect Ratio (EAR) to detect eye closure
        
        Args:
            eye_landmarks: Coordinates of eye landmarks
            
        Returns:
            float: Eye aspect ratio
        """
        if len(eye_landmarks) < 6:
            return 0.0
        
        # Calculate distances between eye landmarks
        A = distance.euclidean(eye_landmarks[1], eye_landmarks[5])
        B = distance.euclidean(eye_landmarks[2], eye_landmarks[4])
        C = distance.euclidean(eye_landmarks[0], eye_landmarks[3])
        
        # Calculate aspect ratio
        ear = (A + B) / (2.0 * C)
        return ear
    
    def detect_drowsiness(self, frame):
        """
        Detect drowsiness in the frame
        
        Args:
            frame: Input video frame
            
        Returns:
            dict: Drowsiness detection results
        """
        h, w, c = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        drowsiness_detected = False
        left_ear = 0.0
        right_ear = 0.0
        drowsiness_level = 0.0
        head_yaw = 0.0
        head_pitch = 0.0
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            landmarks = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks.landmark])
            
            # Get eye landmarks
            left_eye = landmarks[self.LEFT_EYE]
            right_eye = landmarks[self.RIGHT_EYE]
            
            # Calculate eye aspect ratios
            left_ear = self.calculate_eye_aspect_ratio(left_eye)
            right_ear = self.calculate_eye_aspect_ratio(right_eye)
            
            # Average EAR
            ear = (left_ear + right_ear) / 2.0
            self.eye_closure_history.append(ear)
            
            # Check for drowsiness (closed eyes)
            if ear < self.eye_aspect_ratio_threshold:
                self.drowsiness_counter += 1
            else:
                if self.drowsiness_counter >= self.drowsiness_frames_threshold:
                    self.blink_count += 1
                self.drowsiness_counter = 0
            
            # Drowsiness alert if eyes closed too long
            if self.drowsiness_counter >= self.drowsiness_frames_threshold:
                drowsiness_detected = True
                self.drowsiness_alert = True
            else:
                self.drowsiness_alert = False
            
            # Calculate head position (yaw and pitch)
            head_yaw, head_pitch = self.calculate_head_position(landmarks)
            self.head_position_history.append((head_yaw, head_pitch))
            
            # Calculate overall drowsiness level (0-1)
            drowsiness_level = min(1.0, (1 - ear / 0.4) * 1.2)
            drowsiness_level = max(0.0, drowsiness_level)
        
        return {
            'drowsiness_detected': drowsiness_detected,
            'drowsiness_level': drowsiness_level,
            'left_ear': left_ear,
            'right_ear': right_ear,
            'head_yaw': head_yaw,
            'head_pitch': head_pitch,
            'blink_count': self.blink_count,
            'drowsiness_counter': self.drowsiness_counter,
            'alert': self.drowsiness_alert
        }
    
    def calculate_head_position(self, landmarks):
        """
        Calculate head yaw and pitch angles
        
        Args:
            landmarks: Face landmarks
            
        Returns:
            tuple: (yaw, pitch) angles in degrees
        """
        # Key points for head pose estimation
        nose = landmarks[1]  # Nose tip
        left_eye = landmarks[33]  # Left eye
        right_eye = landmarks[263]  # Right eye
        mouth_left = landmarks[61]  # Left mouth
        mouth_right = landmarks[291]  # Right mouth
        
        # Calculate yaw (horizontal head rotation)
        eye_distance = np.linalg.norm(left_eye - right_eye)
        nose_to_left = np.linalg.norm(nose - left_eye)
        nose_to_right = np.linalg.norm(nose - right_eye)
        
        if eye_distance > 0:
            yaw = np.degrees(np.arctan((nose_to_left - nose_to_right) / eye_distance))
        else:
            yaw = 0
        
        # Calculate pitch (vertical head rotation)
        face_height = np.linalg.norm(landmarks[10] - landmarks[152])
        face_width = eye_distance
        
        if face_height > 0:
            pitch = np.degrees(np.arctan(face_width / face_height))
        else:
            pitch = 0
        
        return yaw, pitch
    
    def analyze_blink_rate(self):
        """
        Analyze blink rate to detect abnormal patterns
        
        Returns:
            dict: Blink analysis
        """
        if len(self.eye_closure_history) < 10:
            return {
                'blink_rate': 0,
                'abnormal_blink': False,
                'total_blinks': self.blink_count
            }
        
        # Detect blinks (transitions from open to closed to open)
        blinks_in_window = 0
        eye_closure_array = np.array(list(self.eye_closure_history))
        
        for i in range(1, len(eye_closure_array) - 1):
            if (eye_closure_array[i-1] > self.eye_aspect_ratio_threshold and
                eye_closure_array[i] < self.eye_aspect_ratio_threshold and
                eye_closure_array[i+1] > self.eye_aspect_ratio_threshold):
                blinks_in_window += 1
        
        # Normal blink rate: 15-30 blinks per minute
        blink_rate = blinks_in_window * 2  # Approximate per minute
        abnormal_blink = blink_rate > 40 or blink_rate < 5
        
        return {
            'blink_rate': blink_rate,
            'abnormal_blink': abnormal_blink,
            'total_blinks': self.blink_count
        }
    
    def get_drowsiness_statistics(self):
        """
        Get drowsiness statistics
        
        Returns:
            dict: Drowsiness statistics
        """
        if len(self.eye_closure_history) == 0:
            return {
                'avg_ear': 0.0,
                'min_ear': 0.0,
                'closed_eye_percentage': 0.0,
                'avg_head_yaw': 0.0,
                'avg_head_pitch': 0.0
            }
        
        ear_array = np.array(list(self.eye_closure_history))
        closed_eyes = np.sum(ear_array < self.eye_aspect_ratio_threshold)
        closed_percentage = (closed_eyes / len(ear_array)) * 100
        
        head_positions = np.array(list(self.head_position_history))
        avg_yaw = np.mean(head_positions[:, 0]) if len(head_positions) > 0 else 0
        avg_pitch = np.mean(head_positions[:, 1]) if len(head_positions) > 0 else 0
        
        return {
            'avg_ear': float(np.mean(ear_array)),
            'min_ear': float(np.min(ear_array)),
            'closed_eye_percentage': float(closed_percentage),
            'avg_head_yaw': float(avg_yaw),
            'avg_head_pitch': float(avg_pitch),
            'blink_count': self.blink_count
        }
    
    def draw_drowsiness_info(self, frame, results):
        """
        Draw drowsiness detection info on frame
        
        Args:
            frame: Input frame
            results: Detection results from detect_drowsiness()
            
        Returns:
            frame: Annotated frame
        """
        h, w, c = frame.shape
        
        # Draw alert status
        if results['drowsiness_detected']:
            cv2.putText(frame, "DROWSINESS ALERT!", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.rectangle(frame, (30, 30), (400, 80), (0, 0, 255), 3)
        
        # Draw drowsiness level bar
        bar_width = int(results['drowsiness_level'] * 200)
        cv2.rectangle(frame, (10, 100), (210, 130), (200, 200, 200), 2)
        cv2.rectangle(frame, (10, 100), (10 + bar_width, 130), (0, 165, 255), -1)
        cv2.putText(frame, f"Drowsiness: {results['drowsiness_level']:.2f}", (10, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw head position
        cv2.putText(frame, f"Head Yaw: {results['head_yaw']:.1f}°", (10, 180),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Head Pitch: {results['head_pitch']:.1f}°", (10, 210),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Draw blink count
        cv2.putText(frame, f"Blinks: {results['blink_count']}", (10, 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Draw EAR values
        cv2.putText(frame, f"Left EAR: {results['left_ear']:.2f}", (w - 300, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Right EAR: {results['right_ear']:.2f}", (w - 300, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return frame
