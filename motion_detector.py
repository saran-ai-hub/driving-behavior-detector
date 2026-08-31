import cv2
import numpy as np
from collections import deque
import imutils

class MotionDetector:
    """
    Detects vehicle motion and unusual movement patterns
    """
    def __init__(self, frame_width=640, frame_height=480, motion_threshold=500):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.motion_threshold = motion_threshold
        self.previous_frame = None
        self.motion_history = deque(maxlen=30)
        self.motion_detected = False
        self.motion_intensity = 0.0
        self.movement_vectors = deque(maxlen=10)
        
    def detect_motion(self, frame):
        """
        Detect motion in the frame using frame difference
        
        Args:
            frame: Input video frame
            
        Returns:
            motion_detected (bool): Whether motion is detected
            motion_intensity (float): Intensity of motion (0-1)
            contours: Motion contours
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # Initialize previous frame on first call
        if self.previous_frame is None:
            self.previous_frame = gray
            return False, 0.0, []
        
        # Calculate frame difference
        frame_delta = cv2.absdiff(self.previous_frame, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        
        # Dilate to fill gaps
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Find contours
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        
        # Calculate motion intensity
        motion_area = sum([cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 30])
        total_area = self.frame_width * self.frame_height
        self.motion_intensity = min(1.0, motion_area / total_area * 10)
        
        # Detect motion
        self.motion_detected = motion_area >= self.motion_threshold
        
        # Store motion history
        self.motion_history.append(self.motion_intensity)
        
        self.previous_frame = gray
        return self.motion_detected, self.motion_intensity, contours
    
    def analyze_sudden_movement(self, frame):
        """
        Detect sudden/abrupt movements that indicate accident risk
        
        Args:
            frame: Input video frame
            
        Returns:
            sudden_movement_detected (bool): Whether sudden movement is detected
            movement_risk_score (float): Risk score (0-1)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.previous_frame is None:
            self.previous_frame = gray
            return False, 0.0
        
        # Optical flow for better movement detection
        flow = cv2.calcOpticalFlowFarneback(
            self.previous_frame, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, n8=False, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        # Calculate magnitude and angle
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Calculate average motion magnitude
        avg_magnitude = np.mean(magnitude)
        max_magnitude = np.max(magnitude)
        
        # Detect sudden movements (high magnitude changes)
        sudden_movement = max_magnitude > 20
        
        # Calculate risk score based on movement characteristics
        movement_risk_score = min(1.0, (avg_magnitude / 5 + max_magnitude / 50) / 2)
        
        # Store movement vector
        self.movement_vectors.append(max_magnitude)
        
        self.previous_frame = gray
        return sudden_movement, movement_risk_score
    
    def get_motion_statistics(self):
        """
        Get motion statistics over time window
        
        Returns:
            dict: Motion statistics
        """
        if len(self.motion_history) == 0:
            return {
                'avg_motion': 0.0,
                'max_motion': 0.0,
                'motion_variance': 0.0,
                'total_motion_events': 0
            }
        
        motion_array = np.array(list(self.motion_history))
        motion_events = np.sum(motion_array > 0.1)
        
        return {
            'avg_motion': float(np.mean(motion_array)),
            'max_motion': float(np.max(motion_array)),
            'motion_variance': float(np.var(motion_array)),
            'total_motion_events': int(motion_events),
            'recent_motion': self.motion_detected
        }
    
    def draw_motion_on_frame(self, frame, contours, motion_detected, motion_intensity):
        """
        Draw motion visualization on frame
        
        Args:
            frame: Input frame
            contours: Motion contours
            motion_detected: Motion detection status
            motion_intensity: Motion intensity value
            
        Returns:
            frame: Annotated frame
        """
        # Draw contours
        for contour in contours:
            if cv2.contourArea(contour) > 30:
                (x, y, w, h) = cv2.boundingRect(contour)
                color = (0, 0, 255) if motion_detected else (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Add motion intensity bar
        bar_height = int(motion_intensity * 100)
        cv2.rectangle(frame, (10, 10), (30, 110), (200, 200, 200), 2)
        cv2.rectangle(frame, (10, 110 - bar_height), (30, 110), (0, 255, 0), -1)
        cv2.putText(frame, f"Motion: {motion_intensity:.2f}", (40, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
