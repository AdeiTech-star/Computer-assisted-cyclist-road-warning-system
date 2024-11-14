# src/classes/prediction.py

import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np
import os

class TurnPredictionCNN(nn.Module):
    def __init__(self):
        super(TurnPredictionCNN, self).__init__()
        
        # CNN for processing cropped vehicle images
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(128 + 3, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

class TurnPredictor:
    def __init__(self, model_path=None):
        # Initialize turn prediction CNN
        self.turn_predictor = TurnPredictionCNN()
        if model_path and os.path.exists(model_path):
            self.turn_predictor.load_state_dict(torch.load(model_path))
        self.turn_predictor.eval()

        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.turn_predictor.to(self.device)

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

        # Turn prediction thresholds
        self.TURN_THRESHOLDS = {
            'left': -15,
            'right': 15
        }

    def predict_turn(self, frame, vehicle_data):
        """Predict turn based on vehicle data"""
        bbox = vehicle_data['bbox']
        x1, y1, x2, y2 = bbox
        
        # Extract vehicle image
        vehicle_img = frame[y1:y2, x1:x2]
        if vehicle_img.size == 0:
            return {
                'prediction': 'straight',
                'confidence': 0.33
            }

        # Calculate relative position and angle
        frame_height, frame_width = frame.shape[:2]
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        rel_x = (center_x - frame_width/2) / frame_width
        rel_y = (center_y - frame_height) / frame_height
        angle = vehicle_data['angle']

        try:
            # Prepare image for CNN
            img_tensor = self.transform(vehicle_img)
            img_tensor = img_tensor.unsqueeze(0).to(self.device)

            # Prepare features
            features = torch.tensor([[angle, rel_x, rel_y]], 
                                  dtype=torch.float32).to(self.device)

            # Get prediction
            with torch.no_grad():
                cnn_features = self.turn_predictor.conv_layers(img_tensor)
                cnn_features = cnn_features.view(1, -1)
                combined_features = torch.cat([cnn_features, features], dim=1)
                output = self.turn_predictor.fc_layers(combined_features)
                probabilities = torch.softmax(output, dim=1)
                prediction = torch.argmax(probabilities).item()

                return {
                    'prediction': ['left', 'straight', 'right'][prediction],
                    'confidence': probabilities[0][prediction].item()
                }
        except Exception as e:
            print(f"Error in turn prediction: {e}")
            return {
                'prediction': 'straight',
                'confidence': 0.33
            }