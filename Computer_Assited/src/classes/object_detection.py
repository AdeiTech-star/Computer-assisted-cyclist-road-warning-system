from ultralytics import YOLO
import cv2
import numpy as np

class VehicleDetector:
    def __init__(self):
        # Load YOLO model
        self.model = YOLO('yolov8n.pt')

        # Known parameters
        self.KNOWN_DISTANCES = {
            'car': 1.8,      # Average car width in meterspip 
            'bus': 2.55      # Average bus width in meters
        }

        # Multiple distance thresholds
        self.DISTANCE_THRESHOLDS = {
            'danger': 7.0,    # Red line - immediate danger
            'moderate': 12.0, # Orange line - moderate safety
            'safe': 15.0     # Green line - safe distance
        }

        # Colors for different zones
        self.ZONE_COLORS = {
            'danger': (0, 0, 255),     # Red
            'moderate': (0, 165, 255),  # Orange
            'safe': (0, 255, 0)        # Green
        }

        self.focal_length = None

    def calculate_focal_length(self, known_distance, known_width, pixel_width):
        return (pixel_width * known_distance) / known_width

    def calculate_distance(self, pixel_width, actual_width):
        if pixel_width == 0:
            return float('inf')
        return (actual_width * self.focal_length) / pixel_width

    def get_distance_status(self, distance):
        if distance <= self.DISTANCE_THRESHOLDS['danger']:
            return 'danger'
        elif distance <= self.DISTANCE_THRESHOLDS['moderate']:
            return 'moderate'
        else:
            return 'safe'

    def detect_vehicles(self, frame, calibration_distance=7.0, calibration_width_pixels=300):
        if frame is None:
            print("Error: Invalid frame.")
            return []

        # Calculate focal length if not already calculated
        if self.focal_length is None:
            self.focal_length = self.calculate_focal_length(
                calibration_distance,
                self.KNOWN_DISTANCES['car'],
                calibration_width_pixels
            )

        # Run YOLOv8 detection
        results = self.model(frame)[0]

        # Process detections
        detected_vehicles = []
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = result
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            class_name = results.names[int(class_id)]

            if class_name in self.KNOWN_DISTANCES:
                pixel_width = x2 - x1
                actual_width = self.KNOWN_DISTANCES[class_name]
                distance = self.calculate_distance(pixel_width, actual_width)
                status = self.get_distance_status(distance)

                detected_vehicles.append({
                    'class': class_name,
                    'distance': distance,
                    'confidence': confidence,
                    'bbox': (x1, y1, x2, y2),
                    'status': status
                })

        return detected_vehicles