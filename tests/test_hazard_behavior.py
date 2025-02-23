import numpy as np
import time
from dataclasses import dataclass

# Constants for PWM values
STOP_PWM = 1500
REVERSE_PWM = 1450
MIN_PWM = 1540
STRAIGHT_PWM = 1560
MAX_PWM = 1570

@dataclass
class MockDetection:
    class_id: int
    bbox: np.ndarray

class MockController:
    def __init__(self):
        self.right_pwm = STOP_PWM
        self.left_pwm = STOP_PWM
        self.log = []
        self.start_time = time.time()
    
    def set_servo(self, pin, pwm):
        if pin == 5:
            self.right_pwm = pwm
        elif pin == 6:
            self.left_pwm = pwm
        self.log.append({
            'time': time.time() - self.start_time,
            'pin': pin,
            'pwm': pwm
        })
    
    def get_status(self):
        return {
            'right_pwm': self.right_pwm,
            'left_pwm': self.left_pwm,
            'elapsed': time.time() - self.start_time,
            'log': self.log
        }

def handle_hazard_detection(controller, detections, hazard_detection_time=None, is_reversing=False):
    """Simplified hazard detection and motor control logic"""
    # Process detections
    hazard_buoys = []
    nav_buoys = []
    
    for detection in detections:
        if detection.class_id in [2, 3]:  # Yellow or black buoy
            hazard_buoys.append(detection)
        elif detection.class_id in [0, 1]:  # Red or green buoy
            nav_buoys.append(detection)
    
    current_time = time.time()
    
    # Handle hazard detection
    if hazard_buoys:
        if not is_reversing:
            hazard_detection_time = current_time
            is_reversing = True
        
        # Apply reverse motion
        controller.set_servo(5, REVERSE_PWM)
        controller.set_servo(6, REVERSE_PWM)
        return hazard_detection_time, is_reversing
    
    # Check if we should continue reversing
    if is_reversing and hazard_detection_time is not None:
        elapsed = current_time - hazard_detection_time
        if elapsed < 2.0:
            # Continue reversing during 2-second period
            controller.set_servo(5, REVERSE_PWM)
            controller.set_servo(6, REVERSE_PWM)
            return hazard_detection_time, is_reversing
        else:
            # After 2 seconds, stop reversing and transition to normal navigation
            is_reversing = False
            hazard_detection_time = None
            
            # Normal navigation after reverse period
            if nav_buoys:
                controller.set_servo(5, STRAIGHT_PWM)
                controller.set_servo(6, STRAIGHT_PWM)
            else:
                controller.set_servo(5, STOP_PWM)
                controller.set_servo(6, STOP_PWM)
            return None, False
    
    # Normal navigation (when not reversing)
    if nav_buoys:
        controller.set_servo(5, STRAIGHT_PWM)
        controller.set_servo(6, STRAIGHT_PWM)
    else:
        controller.set_servo(5, STOP_PWM)
        controller.set_servo(6, STOP_PWM)
    
    return hazard_detection_time, is_reversing

def test_hazard_behavior():
    """Test hazard detection and reverse motion behavior"""
    print("Hazard Detection and Reverse Motion Tests:")
    
    test_cases = [
        {
            'name': "Yellow hazard buoy only",
            'detections': [
                MockDetection(
                    class_id=2,  # Yellow buoy
                    bbox=np.array([640, 360, 680, 400])
                )
            ],
            'expected': {
                'initial': {'right_pwm': REVERSE_PWM, 'left_pwm': REVERSE_PWM},
                'check_continuation': False
            }
        },
        {
            'name': "Black hazard buoy only",
            'detections': [
                MockDetection(
                    class_id=3,  # Black buoy
                    bbox=np.array([640, 360, 680, 400])
                )
            ],
            'expected': {
                'initial': {'right_pwm': REVERSE_PWM, 'left_pwm': REVERSE_PWM},
                'check_continuation': False
            }
        },
        {
            'name': "Hazard with navigation buoys",
            'sequence': [
                # First frame: Hazard buoy
                {
                    'detections': [
                        MockDetection(
                            class_id=2,  # Yellow buoy
                            bbox=np.array([640, 360, 680, 400])
                        )
                    ],
                    'expected': {'right_pwm': REVERSE_PWM, 'left_pwm': REVERSE_PWM},
                    'delay': 0.5  # Initial reverse motion
                },
                # Second frame: Navigation buoys appear, continue reversing
                {
                    'detections': [
                        MockDetection(
                            class_id=0,  # Red buoy
                            bbox=np.array([400, 360, 440, 400])
                        ),
                        MockDetection(
                            class_id=1,  # Green buoy
                            bbox=np.array([880, 360, 920, 400])
                        )
                    ],
                    'expected': {'right_pwm': REVERSE_PWM, 'left_pwm': REVERSE_PWM},
                    'delay': 2.0  # Continue reversing for 2 seconds
                },
                # Third frame: After 2 seconds, transition to normal navigation
                {
                    'detections': [
                        MockDetection(
                            class_id=0,  # Red buoy
                            bbox=np.array([400, 360, 440, 400])
                        ),
                        MockDetection(
                            class_id=1,  # Green buoy
                            bbox=np.array([880, 360, 920, 400])
                        )
                    ],
                    'expected': {'right_pwm': STRAIGHT_PWM, 'left_pwm': STRAIGHT_PWM}
                }
            ]
        }
    ]
    
    for test in test_cases:
        print(f"\nTest: {test['name']}")
        controller = MockController()
        hazard_time = None
        is_reversing = False
        
        if 'sequence' in test:
            # Test sequence of frames
            for i, frame in enumerate(test['sequence']):
                print(f"\nFrame {i + 1}:")
                hazard_time, is_reversing = handle_hazard_detection(
                    controller, frame['detections'], hazard_time, is_reversing)
                
                status = controller.get_status()
                print(f"Right motor PWM: {status['right_pwm']} (Expected: {frame['expected']['right_pwm']})")
                print(f"Left motor PWM: {status['left_pwm']} (Expected: {frame['expected']['left_pwm']})")
                
                assert status['right_pwm'] == frame['expected']['right_pwm'], \
                    f"Right motor PWM mismatch: got {status['right_pwm']}, expected {frame['expected']['right_pwm']}"
                assert status['left_pwm'] == frame['expected']['left_pwm'], \
                    f"Left motor PWM mismatch: got {status['left_pwm']}, expected {frame['expected']['left_pwm']}"
                
                if 'delay' in frame:
                    print(f"Waiting {frame['delay']} seconds...")
                    time.sleep(frame['delay'])
        else:
            # Single frame test
            hazard_time, is_reversing = handle_hazard_detection(
                controller, test['detections'], hazard_time, is_reversing)
            
            status = controller.get_status()
            print(f"Right motor PWM: {status['right_pwm']} (Expected: {test['expected']['initial']['right_pwm']})")
            print(f"Left motor PWM: {status['left_pwm']} (Expected: {test['expected']['initial']['left_pwm']})")
            
            assert status['right_pwm'] == test['expected']['initial']['right_pwm'], \
                f"Right motor PWM mismatch: got {status['right_pwm']}, expected {test['expected']['initial']['right_pwm']}"
            assert status['left_pwm'] == test['expected']['initial']['left_pwm'], \
                f"Left motor PWM mismatch: got {status['left_pwm']}, expected {test['expected']['initial']['left_pwm']}"
        
        print("✓ Test passed")

if __name__ == "__main__":
    test_hazard_behavior()
