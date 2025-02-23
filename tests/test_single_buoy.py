import numpy as np
from dataclasses import dataclass

@dataclass
class MockDetection:
    bbox: np.ndarray
    class_id: int
    xyxy: list

class MockController:
    def __init__(self):
        self.right_pwm = 1500  # Pin 5
        self.left_pwm = 1500   # Pin 6
        self.log = []
    
    def set_servo(self, pin, pwm):
        if pin == 5:
            self.right_pwm = pwm
            self.log.append(f"Right motor (pin 5): {pwm}")
        elif pin == 6:
            self.left_pwm = pwm
            self.log.append(f"Left motor (pin 6): {pwm}")
    
    def get_status(self):
        return {
            'right_pwm': self.right_pwm,
            'left_pwm': self.left_pwm
        }

def create_mock_depth_map(width=1280, height=720):
    depth_map = np.zeros((height, width))
    return depth_map

def test_single_buoy_detection():
    # Import only the functions we need to test
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    # Mock the pyzed.sl module
    class MockSL:
        class Mat:
            def __init__(self):
                self.data = None
            
            def get_data(self):
                return self.data
        
        class ERROR_CODE:
            SUCCESS = 0
    
    sys.modules['pyzed.sl'] = MockSL()
    
    from obstacle_avoidance import find_optimal_path
    
    frame_width = 1280
    frame_height = 720
    
    # Test cases
    test_cases = [
        {
            'name': "Only green buoy",
            'detections': [
                MockDetection(
                    bbox=np.array([800, 360, 840, 400]),
                    class_id=1,  # Green buoy
                    xyxy=[np.array([800, 360, 840, 400])]
                )
            ],
            'expected': {
                'action': "turn_left",
                'right_pwm': 1570,
                'left_pwm': 1500
            }
        },
        {
            'name': "Only red buoy",
            'detections': [
                MockDetection(
                    bbox=np.array([400, 360, 440, 400]),
                    class_id=0,  # Red buoy
                    xyxy=[np.array([400, 360, 440, 400])]
                )
            ],
            'expected': {
                'action': "turn_right",
                'right_pwm': 1500,
                'left_pwm': 1570
            }
        }
    ]
    
    print("Single Buoy Detection Tests:")
    for test in test_cases:
        print(f"\nTest: {test['name']}")
        
        # Create mock depth map with 5m depth for buoys
        depth_map = create_mock_depth_map()
        for detection in test['detections']:
            x1, y1, x2, y2 = detection.bbox
            center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
            depth_map[center_y, center_x] = 5.0
        
        # Test path finding
        target_x, target_depth = find_optimal_path(test['detections'], depth_map, frame_width)
        
        if target_x is not None:
            print(f"Target point found at x={target_x:.1f}, depth={target_depth:.1f}m")
        else:
            print("No target point found")
        
        # Test motor control logic
        controller = MockController()
        
        # Simulate motor control based on target point
        if target_x is not None:
            error = target_x - frame_width/2
            
            if error > 0:  # Need to turn right
                controller.set_servo(5, 1500)  # Stop right motor
                controller.set_servo(6, 1570)  # Left motor full
            else:  # Need to turn left
                controller.set_servo(5, 1570)  # Right motor full
                controller.set_servo(6, 1500)  # Stop left motor
        
        status = controller.get_status()
        print(f"Motor PWM values:")
        print(f"Right motor: {status['right_pwm']} (Expected: {test['expected']['right_pwm']})")
        print(f"Left motor: {status['left_pwm']} (Expected: {test['expected']['left_pwm']})")
        
        # Verify PWM values
        assert status['right_pwm'] == test['expected']['right_pwm'], \
            f"Right motor PWM mismatch: got {status['right_pwm']}, expected {test['expected']['right_pwm']}"
        assert status['left_pwm'] == test['expected']['left_pwm'], \
            f"Left motor PWM mismatch: got {status['left_pwm']}, expected {test['expected']['left_pwm']}"
        
        print("✓ Test passed")

if __name__ == "__main__":
    test_single_buoy_detection()
