# DevinLite

Autonomous USV (Unmanned Surface Vehicle) control system for RoboBoat 2025 competition using ZED2i camera and YOLO object detection.

## Features

- Real-time buoy detection and classification
- Autonomous navigation between red and green buoys
- Hazard avoidance (yellow and black buoys)
- Multiple navigation scenarios:
  - Single buoy detection (red/green)
  - Hazard buoy detection
  - No-detection search pattern
  - Triangle-based path planning

## Requirements

- ZED2i camera
- CUDA 12.6
- A100 GPU
- Python dependencies:
  - pyzed
  - opencv-python
  - numpy
  - ultralytics (YOLO)
  - supervision

## Motor Control

PWM values for different motions:
- Stop: 1500 PWM
- Forward (straight): 1560 PWM
- Turn right: Right motor 1500 PWM, Left motor 1560-1570 PWM
- Turn left: Left motor 1500 PWM, Right motor 1560-1570 PWM
- Reverse: 1450 PWM

## Usage

Run the main script:
```bash
python main.py
```

The script will initialize the ZED camera, load the YOLO model, and begin autonomous navigation.
