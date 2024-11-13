# Computer-Assisted Cyclist Road Safety Warning System

## Overview
This research presents a real-time warning system designed to enhance cyclist safety in urban environments. The system uses monocular vision and deep learning techniques to detect and assess potential threats from approaching vehicles.

## Key Features
- Real-time vehicle detection using YOLOv8
- Distance estimation through Inverse Perspective Mapping (IPM)
- Vehicle orientation tracking and threat assessment
- Bird's-eye view transformation for improved spatial awareness
- Multi-vehicle tracking and movement prediction

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

## Repository Structure
- `/src` - Source code for the warning system
- `/data` - Test datasets and results
- `/docs` - Additional documentation and research paper
- `/models` - Trained models and weights

## Citation
If you use this work in your research, please cite:
```
@article{thema2024computer,
  title={Computer-assisted cyclist road safety warning system},
  author={Thema, Nthabiseng and Bau, Hairong},
  institution={University of the Witwatersrand},
  year={2024}
}
```
