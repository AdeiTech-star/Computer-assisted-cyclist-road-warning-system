# src/classes/angle.py

import cv2
import numpy as np
from ultralytics import YOLO

class AngleDetector:
    def __init__(self):
        # Load YOLO model
        self.model = YOLO('yolov8n.pt')

        # Vehicle classes in COCO dataset
        self.vehicle_classes = {
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck'
        }

        # Configuration parameters
        self.conf_threshold = 0.5

        # Angle thresholds
        self.ANGLE_THRESHOLDS = {
            'overtaking_left': -45,
            'overtaking_right': 45,
            'warning_left': -20,
            'warning_right': 20
        }

        # Status colors
        self.COLORS = {
            'overtaking': (0, 0, 255),    # Red
            'warning': (0, 165, 255),     # Orange
            'safe': (0, 255, 0)           # Green
        }

    def calculate_vehicle_angle(self, frame, vehicle_box):
        """Calculate angle between detected vehicle and camera center"""
        frame_height, frame_width = frame.shape[:2]
        x, y, w, h = vehicle_box

        # Calculate center points
        vehicle_center_x = x + w/2
        vehicle_center_y = y + h/2

        # Calculate relative position to our vehicle
        our_center_x = frame_width/2
        our_center_y = frame_height

        # Calculate angle using arctangent
        dx = vehicle_center_x - our_center_x
        dy = our_center_y - vehicle_center_y
        angle = np.degrees(np.arctan2(dx, dy))

        # Determine position
        if angle < -10:
            position = "left"
        elif angle > 10:
            position = "right"
        else:
            position = "center"

        return angle, position, vehicle_center_y

    def get_status(self, angle, position_y, frame_height):
        """Determine vehicle status based on angle and position"""
        relative_position = position_y / frame_height

        if abs(angle) > self.ANGLE_THRESHOLDS['overtaking_left']:
            return 'overtaking'
        elif abs(angle) > self.ANGLE_THRESHOLDS['warning_left'] and relative_position > 0.4:
            return 'warning'
        return 'safe'