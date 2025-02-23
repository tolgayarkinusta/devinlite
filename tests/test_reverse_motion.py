import numpy as np
import time
from dataclasses import dataclass

@dataclass
class MockDetection:
    bbox: np.ndarray
    class_id: int
    xyxy: list

class MockController:
    def __init__(self):
        self.right_pwm = 1500
        self.left_pwm = 1500
        self.log = []
        self.last_command_time = None
    
    def set_servo(self, pin, pwm):
        if self.last_command_time is None:
            self.last_command_time = time.time()
        
        if pin == 5:
            self.right_pwm = pwm
            self.log.append(f"Right motor (pin 5): {pwm} at t={time.time() - self.last_command_time:.1f}s")
        elif pin == 6:
            self.left_pwm = pwm
            self.log.append(f"Left motor (pin 6): {pwm} at t={time.time() - self.last_command_time:.1f}s")
    
    def get_status(self):
        return {
            'right_pwm': self.right_pwm,
            'left_pwm': self.left_pwm,
            'time_elapsed': time.time() - (self.last_command_time or time.time())
        }

def create_mock_depth_map(width=1280, height=720):
    depth_map = np.zeros((height, width))
    return depth_map

def test_reverse_motion():
    """Test reverse motion behavior with hazard buoys"""
    # Mock ZED camera dependencies
    class MockSL:
        class Mat:
            def __init__(self):
                self.data = None
            def get_data(self):
                return self.data
        
        class RuntimeParameters:
            pass
        
        class VIEW:
            LEFT = 0
        
        class MEASURE:
            DEPTH = 0
        
        class ERROR_CODE:
            SUCCESS = 0
    
    # Mock YOLO model
    class MockModel:
        def __call__(self, frame):
            return [None]
    
    # Import and patch modules
    import sys
    sys.modules['pyzed.sl'] = MockSL()
    
    from obstacle_avoidance import navigate_usv
    
    frame_width = 1280
    frame_height = 720
    controller = MockController()
    
    test_cases = [
        {
            'name': "Yellow hazard buoy only",
            'detections': [
                MockDetection(
                    bbox=np.array([640, 360, 680, 400]),
                    class_id=2,  # Yellow buoy
                    xyxy=[np.array([640, 360, 680, 400])]
                )
            ],
            'expected': {
                'action': "reverse",
                'right_pwm': 1450,
                'left_pwm': 1450
            }
        },
        {
            'name': "Black hazard buoy only",
            'detections': [
                MockDetection(
                    bbox=np.array([640, 360, 680, 400]),
                    class_id=3,  # Black buoy
                    xyxy=[np.array([640, 360, 680, 400])]
                )
            ],
            'expected': {
                'action': "reverse",
                'right_pwm': 1450,
                'left_pwm': 1450
            }
        },
        {
            'name': "Hazard buoy with navigation buoys",
            'detections': [
                MockDetection(
                    bbox=np.array([640, 360, 680, 400]),
                    class_id=2,  # Yellow buoy
                    xyxy=[np.array([640, 360, 680, 400])]
                ),
                MockDetection(
                    bbox=np.array([400, 360, 440, 400]),
                    class_id=0,  # Red buoy
                    xyxy=[np.array([400, 360, 440, 400])]
                ),
                MockDetection(
                    bbox=np.array([880, 360, 920, 400]),
                    class_id=1,  # Green buoy
                    xyxy=[np.array([880, 360, 920, 400])]
                )
            ],
            'expected': {
                'initial': {'right_pwm': 1450, 'left_pwm': 1450},
                'duration': 2.0,
                'final': {'right_pwm': 1560, 'left_pwm': 1560}  # Should return to normal navigation
            }
        }
    ]
    
    print("Reverse Motion Tests:")
    for test in test_cases:
        print(f"\nTest: {test['name']}")
        
        # Create mock depth map
        depth_map = create_mock_depth_map(frame_width, frame_height)
        for detection in test['detections']:
            x1, y1, x2, y2 = detection.bbox
            center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
            depth_map[center_y, center_x] = 5.0
        
        # Create mock ZED camera and model
        zed = MockSL()
        model = MockModel()
        
        # Test navigation behavior
        navigate_usv(zed, model, controller)
        status = controller.get_status()
        
        if 'duration' in test['expected']:
            # Test with timing
            print("Initial state:")
            print(f"Right motor PWM: {status['right_pwm']} (Expected: {test['expected']['initial']['right_pwm']})")
            print(f"Left motor PWM: {status['left_pwm']} (Expected: {test['expected']['initial']['left_pwm']})")
            
            # Wait for 2 seconds
            time.sleep(2.0)
            
            # Check final state
            status = controller.get_status()
            print("\nAfter 2 seconds:")
            print(f"Right motor PWM: {status['right_pwm']} (Expected: {test['expected']['final']['right_pwm']})")
            print(f"Left motor PWM: {status['left_pwm']} (Expected: {test['expected']['final']['left_pwm']})")
            
            # Verify timing
            assert status['time_elapsed'] >= 2.0, f"Reverse duration too short: {status['time_elapsed']:.1f}s"
        else:
            # Simple PWM verification
            print(f"Right motor PWM: {status['right_pwm']} (Expected: {test['expected']['right_pwm']})")
            print(f"Left motor PWM: {status['left_pwm']} (Expected: {test['expected']['left_pwm']})")
            
            assert status['right_pwm'] == test['expected']['right_pwm'], \
                f"Right motor PWM mismatch: got {status['right_pwm']}, expected {test['expected']['right_pwm']}"
            assert status['left_pwm'] == test['expected']['left_pwm'], \
                f"Left motor PWM mismatch: got {status['left_pwm']}, expected {test['expected']['left_pwm']}"
        
        print("✓ Test passed")
        
        # Reset controller for next test
        controller = MockController()

if __name__ == "__main__":
    test_reverse_motion()
