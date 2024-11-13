# Computer-Assisted Cyclist Road Safety Warning System

## Overview
This research presents a real-time warning system designed to enhance cyclist safety in urban environments. The system uses monocular vision and deep learning techniques to detect and assess potential threats from approaching vehicles.

## Key Features
- Real-time vehicle detection using YOLOv8
- Distance estimation through Inverse Perspective Mapping (IPM)
- Vehicle orientation tracking and threat assessment
- Bird's-eye view transformation for improved spatial awareness
- Multi-vehicle tracking and movement prediction

## Tech Stack

-Programming Language: Python 3.8+
-Deep Learning Framework: PyTorch
-Computer Vision: OpenCV
-Data Processing: NumPy, Pandas
-Model Training: CUDA (GPU acceleration)

## Libraries and Dependencies

-pytorch >= 1.12.0
-opencv-python >= 4.7.0
-numpy >= 1.21.0
-pandas >= 1.4.0
-ultralytics  # for YOLOv8
-scikit-learn >= 1.0.0
-matplotlib >= 3.5.0


## Technical Implementation

- Object detection: YOLOv8
- Distance estimation: Monocular vision with IPM
- Angle detection: Geometric computation with reference point system
- Warning system: Real-time threat classification (safe, warning, danger)

## Authors
- Nthabiseng Thema - University of the Witwatersrand
- Prof. Hairong Bau - University of the Witwatersrand

## Contact
nthabiseng.thema1@students.wits.ac.za

Repository Structure

/src - Source code for the warning system

/detection - YOLOv8 implementation and vehicle detection
/ipm - Inverse Perspective Mapping algorithms
/tracking - Vehicle tracking and movement prediction
/utils - Helper functions and utilities


/data - Test datasets and results
/docs - Additional documentation and research paper
/models - Trained models and weights
/configs - Configuration files
/tests - Unit tests and integration tests


