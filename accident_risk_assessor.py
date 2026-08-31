import cv2
import numpy as np
from collections import deque
import math

class AccidentRiskAssessor:
    """
    Assesses accident risk based on driving behavior, motion, and drowsiness
    """
    def __init__(self):
        self.risk_history = deque(maxlen=60)
        self.critical_events = deque(maxlen=20)
        self.risk_level = 0.0
        self.risk_category = "LOW"
        self.risk_factors = {}
        
    def assess_risk(self, motion_data, drowsiness_data, vehicle_dynamics=None):
        """
        Calculate overall accident risk based on multiple factors
        
        Args:
            motion_data: Dict with motion detection results
            drowsiness_data: Dict with drowsiness detection results
            vehicle_dynamics: Dict with vehicle speed, acceleration (optional)
            
        Returns:
            dict: Risk assessment results
        """
        risk_factors = {}
        
        # Factor 1: Sudden Movement Risk (20% weight)
        sudden_movement_risk = motion_data.get('sudden_movement_risk', 0.0)
        risk_factors['sudden_movement'] = sudden_movement_risk * 0.20
        
        # Factor 2: Motion Intensity Risk (15% weight)
        motion_intensity = motion_data.get('motion_intensity', 0.0)
        motion_intensity_risk = min(1.0, motion_intensity * 1.5)
        risk_factors['motion_intensity'] = motion_intensity_risk * 0.15
        
        # Factor 3: Drowsiness Risk (35% weight) - Highest weight
        drowsiness_level = drowsiness_data.get('drowsiness_level', 0.0)
        drowsiness_detected = drowsiness_data.get('drowsiness_detected', False)
        drowsiness_risk = drowsiness_level
        if drowsiness_detected:
            drowsiness_risk = min(1.0, drowsiness_level + 0.3)
        risk_factors['drowsiness'] = drowsiness_risk * 0.35
        
        # Factor 4: Head Position Risk (15% weight)
        head_yaw = abs(drowsiness_data.get('head_yaw', 0.0))
        head_pitch = abs(drowsiness_data.get('head_pitch', 0.0))
        head_position_risk = min(1.0, (head_yaw / 45 + head_pitch / 30) / 2)
        risk_factors['head_position'] = head_position_risk * 0.15
        
        # Factor 5: Vehicle Dynamics Risk (15% weight)
        dynamics_risk = 0.0
        if vehicle_dynamics:
            acceleration = vehicle_dynamics.get('acceleration', 0.0)
            speed = vehicle_dynamics.get('speed', 0.0)
            harsh_braking = vehicle_dynamics.get('harsh_braking', False)
            
            acceleration_risk = min(1.0, abs(acceleration) / 5.0)
            speed_risk = min(1.0, speed / 120.0)  # Assuming max speed 120 km/h
            braking_risk = 0.8 if harsh_braking else 0.0
            
            dynamics_risk = max(acceleration_risk, speed_risk, braking_risk)
        risk_factors['vehicle_dynamics'] = dynamics_risk * 0.15
        
        # Calculate total risk
        self.risk_level = sum(risk_factors.values())
        self.risk_level = min(1.0, self.risk_level)
        self.risk_factors = risk_factors
        
        # Determine risk category
        self.risk_category = self._categorize_risk(self.risk_level)
        
        # Store risk history
        self.risk_history.append(self.risk_level)
        
        # Detect critical events
        if self.risk_level > 0.7:
            self.critical_events.append({
                'risk_level': self.risk_level,
                'timestamp': len(self.risk_history),
                'factors': risk_factors.copy()
            })
        
        return {
            'risk_level': self.risk_level,
            'risk_category': self.risk_category,
            'risk_factors': risk_factors,
            'critical_event': self.risk_level > 0.7,
            'recommendations': self._get_recommendations()
        }
    
    def _categorize_risk(self, risk_level):
        """
        Categorize risk level
        
        Args:
            risk_level: Risk level (0-1)
            
        Returns:
            str: Risk category
        """
        if risk_level < 0.3:
            return "LOW"
        elif risk_level < 0.5:
            return "MODERATE"
        elif risk_level < 0.7:
            return "HIGH"
        else:
            return "CRITICAL"
    
    def _get_recommendations(self):
        """
        Get recommendations based on risk factors
        
        Returns:
            list: Safety recommendations
        """
        recommendations = []
        
        if self.risk_factors.get('drowsiness', 0) > 0.25:
            recommendations.append("Pull over and take a break - Driver may be drowsy")
        
        if self.risk_factors.get('sudden_movement', 0) > 0.25:
            recommendations.append("Reduce speed - Sudden movements detected")
        
        if self.risk_factors.get('head_position', 0) > 0.2:
            recommendations.append("Focus on the road - Unusual head position")
        
        if self.risk_factors.get('motion_intensity', 0) > 0.2:
            recommendations.append("Drive carefully - Erratic driving detected")
        
        if self.risk_level > 0.7:
            recommendations.append("EMERGENCY: Stop the vehicle immediately")
        
        return recommendations
    
    def detect_harsh_maneuvers(self, motion_data, previous_motion_intensity=0.0):
        """
        Detect harsh driving maneuvers
        
        Args:
            motion_data: Current motion data
            previous_motion_intensity: Previous frame's motion intensity
            
        Returns:
            dict: Harsh maneuver detection
        """
        current_intensity = motion_data.get('motion_intensity', 0.0)
        intensity_change = abs(current_intensity - previous_motion_intensity)
        
        # Detect sudden acceleration/deceleration
        sudden_change = intensity_change > 0.5
        
        # Detect swerving (lateral movement)
        is_swerving = motion_data.get('sudden_movement_detected', False)
        
        harsh_maneuver_score = min(1.0, intensity_change + (0.5 if is_swerving else 0))
        
        return {
            'harsh_maneuver_detected': sudden_change or is_swerving,
            'maneuver_severity': harsh_maneuver_score,
            'maneuver_type': 'swerve' if is_swerving else 'sudden_acceleration',
            'intensity_change': intensity_change
        }
    
    def analyze_risk_trend(self):
        """
        Analyze risk trend over time
        
        Returns:
            dict: Risk trend analysis
        """
        if len(self.risk_history) < 5:
            return {
                'trend': 'INSUFFICIENT_DATA',
                'trend_direction': 'STABLE',
                'average_risk': 0.0,
                'risk_std': 0.0,
                'peak_risk': 0.0
            }
        
        risk_array = np.array(list(self.risk_history))
        
        # Calculate trend
        if len(risk_array) >= 10:
            recent_avg = np.mean(risk_array[-10:])
            older_avg = np.mean(risk_array[:-10])
            
            if recent_avg > older_avg + 0.1:
                trend_direction = "INCREASING"
            elif recent_avg < older_avg - 0.1:
                trend_direction = "DECREASING"
            else:
                trend_direction = "STABLE"
        else:
            trend_direction = "STABLE"
        
        return {
            'trend': 'ANALYZED',
            'trend_direction': trend_direction,
            'average_risk': float(np.mean(risk_array)),
            'risk_std': float(np.std(risk_array)),
            'peak_risk': float(np.max(risk_array)),
            'min_risk': float(np.min(risk_array))
        }
    
    def get_safety_score(self):
        """
        Calculate overall safety score (inverse of risk)
        
        Returns:
            float: Safety score (0-100)
        """
        if len(self.risk_history) == 0:
            return 100.0
        
        avg_risk = np.mean(list(self.risk_history))
        safety_score = (1.0 - avg_risk) * 100
        return max(0.0, min(100.0, safety_score))
    
    def get_statistics(self):
        """
        Get accident risk statistics
        
        Returns:
            dict: Statistics
        """
        if len(self.risk_history) == 0:
            return {
                'current_risk': 0.0,
                'average_risk': 0.0,
                'max_risk': 0.0,
                'critical_events_count': 0,
                'safety_score': 100.0
            }
        
        risk_array = np.array(list(self.risk_history))
        
        return {
            'current_risk': float(risk_array[-1]),
            'average_risk': float(np.mean(risk_array)),
            'max_risk': float(np.max(risk_array)),
            'critical_events_count': len(self.critical_events),
            'safety_score': self.get_safety_score()
        }
    
    def draw_risk_info(self, frame, risk_data):
        """
        Draw risk assessment info on frame
        
        Args:
            frame: Input frame
            risk_data: Risk assessment results
            
        Returns:
            frame: Annotated frame
        """
        h, w, c = frame.shape
        
        risk_level = risk_data['risk_level']
        risk_category = risk_data['risk_category']
        
        # Draw risk level bar
        bar_width = int(risk_level * 300)
        
        # Color based on risk level
        if risk_category == "LOW":
            color = (0, 255, 0)
        elif risk_category == "MODERATE":
            color = (0, 255, 255)
        elif risk_category == "HIGH":
            color = (0, 165, 255)
        else:  # CRITICAL
            color = (0, 0, 255)
        
        cv2.rectangle(frame, (10, 10), (310, 40), (200, 200, 200), 2)
        cv2.rectangle(frame, (10, 10), (10 + bar_width, 40), color, -1)
        cv2.putText(frame, f"Risk Level: {risk_category} ({risk_level:.2f})", (320, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Draw risk factors
        y_offset = 70
        cv2.putText(frame, "Risk Factors:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        factors = risk_data['risk_factors']
        for i, (factor_name, factor_value) in enumerate(factors.items()):
            text = f"  {factor_name}: {factor_value:.2f}"
            cv2.putText(frame, text, (20, y_offset + 25 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Draw recommendations
        recommendations = risk_data['recommendations']
        if recommendations:
            y_offset = h - 120
            cv2.putText(frame, "Recommendations:", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            for i, rec in enumerate(recommendations[:3]):  # Show top 3
                cv2.putText(frame, f"  • {rec}", (20, y_offset + 25 + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Draw critical event warning
        if risk_data['critical_event']:
            cv2.putText(frame, "!!! CRITICAL EVENT !!!", (w - 400, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
        return frame
