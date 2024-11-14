# src/main.py

import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import torch
import torch.nn as nn
from torchvision import transforms
from classes.object_detection import VehicleDetector
from classes.ipm import IPM
from classes.angle import AngleDetector
from classes.prediction import TurnPredictor

class VehicleSafetySystem:
    def __init__(self):
        self.detector = VehicleDetector()
        self.mapper = IPM()
        self.angle_detector = AngleDetector()
        self.turn_predictor = TurnPredictor()

    def draw_distance_lines(self, frame):
        """Draw distance threshold lines"""
        height, width = frame.shape[:2]
        base_y = int(height * 0.65)

        # Draw threshold lines
        for zone, distance in self.detector.DISTANCE_THRESHOLDS.items():
            y_pos = base_y - distance * 10
            color = self.detector.ZONE_COLORS[zone]
            cv2.line(frame, (0, int(y_pos)), (width, int(y_pos)), color, 2)
            cv2.putText(frame, f"{zone}: {distance}m", (10, int(y_pos) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def draw_detection_and_ipm(self, frame, birds_eye_view, vehicle):
        """Draw detection box and IPM point"""
        x1, y1, x2, y2 = vehicle['bbox']
        color = self.detector.ZONE_COLORS[vehicle['status']]
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw text with background
        text = f"{vehicle['class']}: {vehicle['distance']:.1f}m"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        
        cv2.rectangle(frame, 
                     (x1, y1 - 25), 
                     (x1 + text_size[0], y1),
                     color, -1)
        cv2.putText(frame, text, (x1, y1 - 7),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Transform to bird's eye view
        bottom_center = ((x1 + x2) // 2, y2)
        transformed_point = self.mapper.transform_point(bottom_center)
        cv2.circle(birds_eye_view, transformed_point, 5, color, -1)

    def draw_angle_visualization(self, frame, vehicle):
        """Draw angle visualization"""
        x1, y1, x2, y2 = vehicle['bbox']
        status = vehicle['angle_status']
        angle = vehicle['angle']
        color = self.angle_detector.COLORS[status]
        
        angle_vis = frame.copy()
        
        # Draw center reference line
        cv2.line(angle_vis,
                (frame.shape[1]//2, frame.shape[0]),
                (frame.shape[1]//2, 0),
                (255, 255, 255), 2)
        
        # Draw angle line
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        cv2.line(angle_vis,
                (frame.shape[1]//2, frame.shape[0]),
                (center_x, center_y),
                color, 2)

        # Draw angle text
        text = f"Angle: {angle:.1f}° ({status})"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        
        cv2.rectangle(angle_vis, 
                     (x1, y1 - 25), 
                     (x1 + text_size[0], y1),
                     color, -1)
        cv2.putText(angle_vis, text, (x1, y1 - 7),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return angle_vis

    def draw_turn_prediction(self, frame, vehicle):
        """Draw turn prediction visualization"""
        x1, y1, x2, y2 = vehicle['bbox']
        prediction = vehicle['prediction']
        confidence = vehicle.get('confidence', 0.33)
        
        color = {
            'left': (255, 0, 0),    # Blue
            'right': (0, 0, 255),   # Red
            'straight': (0, 255, 0)  # Green
        }[prediction]
        
        pred_vis = frame.copy()
        
        # Draw bounding box
        cv2.rectangle(pred_vis, (x1, y1), (x2, y2), color, 2)
        
        # Draw text with background
        text = f"Turn: {prediction.upper()} (conf: {confidence:.2f})"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        
        cv2.rectangle(pred_vis, 
                     (x1, y1 - 25), 
                     (x1 + text_size[0], y1),
                     color, -1)
        cv2.putText(pred_vis, text, (x1, y1 - 7),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return pred_vis

    def process_image(self, image_path):
        # Read image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not open image at {image_path}")
            return

        # Get vehicle detections
        detected_vehicles = self.detector.detect_vehicles(frame)
        
        # Get bird's eye view
        birds_eye_view, transform_matrix = self.mapper.get_birds_eye_view(frame)

        # Create copies for different visualizations
        detection_frame = frame.copy()
        angle_frame = None
        prediction_frame = None

        # Draw distance threshold lines on detection frame
        self.draw_distance_lines(detection_frame)

        # Create info text for all vehicles
        info_text = ""
        for i, vehicle in enumerate(detected_vehicles, 1):
            # Calculate angle
            bbox = vehicle['bbox']
            angle, position, center_y = self.angle_detector.calculate_vehicle_angle(
                frame, 
                (bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1])
            )
            
            # Add angle information
            vehicle['angle'] = angle
            vehicle['angle_status'] = self.angle_detector.get_status(
                angle, 
                center_y, 
                frame.shape[0]
            )
            
            # Predict turn
            turn_info = self.turn_predictor.predict_turn(frame, vehicle)
            vehicle.update(turn_info)
            
            # Draw visualizations
            self.draw_detection_and_ipm(detection_frame, birds_eye_view, vehicle)
            
            current_angle_vis = self.draw_angle_visualization(frame, vehicle)
            if angle_frame is None:
                angle_frame = current_angle_vis
            
            current_pred_vis = self.draw_turn_prediction(frame, vehicle)
            if prediction_frame is None:
                prediction_frame = current_pred_vis

            info_text += (f"Vehicle {i}: {vehicle['class']} | "
                         f"Distance: {vehicle['distance']:.1f}m | "
                         f"Angle: {vehicle['angle']:.1f}° | "
                         f"Status: {vehicle['status']} | "
                         f"Turn: {vehicle['prediction']} "
                         f"(conf: {vehicle.get('confidence', 0.33):.2f})\n")

        # Create figure with custom layout
        fig = plt.figure(figsize=(15, 12))
        
        # Add text at the top
        plt.figtext(0.05, 0.95, info_text, fontsize=10, va='top', family='monospace')

        # Plot 1: Detection and IPM (top-left)
        plt.subplot(221)
        plt.imshow(cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB))
        plt.title('Vehicle Detection')
        plt.axis('off')

        # Plot 2: Bird's Eye View (top-right)
        plt.subplot(222)
        plt.imshow(cv2.cvtColor(birds_eye_view, cv2.COLOR_BGR2RGB))
        plt.title("Bird's Eye View")
        plt.axis('off')

        # Plot 3: Angle Visualization (bottom-left)
        plt.subplot(223)
        if angle_frame is not None:
            plt.imshow(cv2.cvtColor(angle_frame, cv2.COLOR_BGR2RGB))
        plt.title('Angle Analysis')
        plt.axis('off')

        # Plot 4: Turn Prediction (bottom-right)
        plt.subplot(224)
        if prediction_frame is not None:
            plt.imshow(cv2.cvtColor(prediction_frame, cv2.COLOR_BGR2RGB))
        plt.title('Turn Prediction')
        plt.axis('off')

        # Adjust layout
        plt.subplots_adjust(top=0.85)
        plt.tight_layout(rect=[0, 0, 1, 0.90])
        plt.show()

        return detected_vehicles

def main():
    system = VehicleSafetySystem()
    image_path = 'data/back_3.jpg'
    detected_vehicles = system.process_image(image_path)

if __name__ == "__main__":
    main()