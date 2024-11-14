# src/utils/image_saver.py
import cv2
import os

class ImageSaver:
    @staticmethod
    def save_results(original_image_path, detection_image, birds_eye_image):
        # Create output directory if it doesn't exist
        output_dir = "output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Get base filename without extension
        base_filename = os.path.basename(original_image_path)
        base_name = os.path.splitext(base_filename)[0]
        
        # Save both images
        detection_path = os.path.join(output_dir, f"{base_name}_detection.jpg")
        birds_eye_path = os.path.join(output_dir, f"{base_name}_birds_eye.jpg")
        
        cv2.imwrite(detection_path, detection_image)
        cv2.imwrite(birds_eye_path, birds_eye_image)
        
        return detection_path, birds_eye_path