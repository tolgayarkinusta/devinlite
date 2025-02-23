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
        self.last_detection_time = time.time()
    
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

def handle_no_detection(controller, detections, last_detection_time, is_turning=False, is_reversing=False):
    """Handle scenario when no buoys are detected"""
    current_time = time.time()
    
    # Check if we have any detections
    if detections:
        # Verify orientation (green buoys should be on right of red buoys)
        red_x = None
        green_x = None
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            center_x = (x1 + x2) / 2
            if detection.class_id == 0:  # Red buoy
                red_x = center_x
            elif detection.class_id == 1:  # Green buoy
                green_x = center_x
        
        # If we have both buoys, check orientation
        if red_x is not None and green_x is not None:
            if green_x > red_x:  # Correct orientation
                controller.set_servo(5, STRAIGHT_PWM)
                controller.set_servo(6, STRAIGHT_PWM)
                return current_time, False, False
            elif is_turning:  # Wrong orientation during turn
                controller.set_servo(5, MIN_PWM)  # Right motor forward
                controller.set_servo(6, REVERSE_PWM)  # Left motor reverse
                return last_detection_time, True, False
        
        # If orientation not verified or incomplete detection
        if is_turning:
            controller.set_servo(5, MIN_PWM)  # Right motor forward
            controller.set_servo(6, REVERSE_PWM)  # Left motor reverse
            return last_detection_time, True, False
        
        return current_time, False, False
    
    # No detections - check timeout
    if current_time - last_detection_time > 2.5:
        if not is_reversing and not is_turning:
            # Start reverse motion
            controller.set_servo(5, REVERSE_PWM)
            controller.set_servo(6, REVERSE_PWM)
            return last_detection_time, False, True
        elif is_reversing:
            # After 2 seconds of reverse, start turning
            if current_time - last_detection_time > 4.5:  # 2.5s timeout + 2s reverse
                controller.set_servo(5, MIN_PWM)  # Right motor forward
                controller.set_servo(6, REVERSE_PWM)  # Left motor reverse
                return last_detection_time, True, False
            else:
                # Continue reversing
                controller.set_servo(5, REVERSE_PWM)
                controller.set_servo(6, REVERSE_PWM)
                return last_detection_time, False, True
        elif is_turning:
            # Continue turning
            controller.set_servo(5, MIN_PWM)  # Right motor forward
            controller.set_servo(6, REVERSE_PWM)  # Left motor reverse
            return last_detection_time, True, False
    
    return last_detection_time, is_turning, is_reversing

def test_no_detection_behavior():
    """Test behavior when no buoys are detected"""
    print("No Detection Behavior Tests:")
    
    # Track state between frames
    last_detection = time.time()
    is_turning = False
    is_reversing = False
    
    test_cases = [
        {
            'name': "No detection timeout and turn sequence",
            'sequence': [
                # Frame 1: Initial state - no detections
                {
                    'detections': [],
                    'expected': {'right_pwm': STOP_PWM, 'left_pwm': STOP_PWM},
                    'delay': 2.6  # Wait for timeout (2.5s + buffer)
                },
                # Frame 2: Start reverse motion
                {
                    'detections': [],
                    'expected': {'right_pwm': REVERSE_PWM, 'left_pwm': REVERSE_PWM},
                    'delay': 2.0  # Reverse for 2 seconds
                },
                # Frame 3: Begin turn-around
                {
                    'detections': [],
                    'expected': {'right_pwm': MIN_PWM, 'left_pwm': REVERSE_PWM}
                },
                # Fourth frame: Continue turning until proper orientation
                {
                    'detections': [
                        MockDetection(
                            class_id=1,  # Green buoy on left (wrong orientation)
                            bbox=np.array([400, 360, 440, 400])
                        ),
                        MockDetection(
                            class_id=0,  # Red buoy on right
                            bbox=np.array([880, 360, 920, 400])
                        )
                    ],
                    'expected': {'right_pwm': MIN_PWM, 'left_pwm': REVERSE_PWM}  # Keep turning
                },
                # Fifth frame: Stop turning when orientation is correct
                {
                    'detections': [
                        MockDetection(
                            class_id=0,  # Red buoy on left
                            bbox=np.array([400, 360, 440, 400])
                        ),
                        MockDetection(
                            class_id=1,  # Green buoy on right
                            bbox=np.array([880, 360, 920, 400])
                        )
                    ],
                    'expected': {'right_pwm': STRAIGHT_PWM, 'left_pwm': STRAIGHT_PWM}  # Resume normal
                }
            ]
        },
        {
            'name': "Detection during turn",
            'sequence': [
                # Start in turning state
                {
                    'detections': [],
                    'is_turning': True,
                    'expected': {'right_pwm': MIN_PWM, 'left_pwm': REVERSE_PWM}
                },
                # Detect buoys but wrong orientation
                {
                    'detections': [
                        MockDetection(
                            class_id=1,  # Green buoy
                            bbox=np.array([400, 360, 440, 400])  # Left side
                        ),
                        MockDetection(
                            class_id=0,  # Red buoy
                            bbox=np.array([880, 360, 920, 400])  # Right side
                        )
                    ],
                    'expected': {'right_pwm': MIN_PWM, 'left_pwm': REVERSE_PWM}  # Continue turning
                },
                # Correct orientation detected
                {
                    'detections': [
                        MockDetection(
                            class_id=0,  # Red buoy
                            bbox=np.array([400, 360, 440, 400])  # Left side
                        ),
                        MockDetection(
                            class_id=1,  # Green buoy
                            bbox=np.array([880, 360, 920, 400])  # Right side
                        )
                    ],
                    'expected': {'right_pwm': STRAIGHT_PWM, 'left_pwm': STRAIGHT_PWM}  # Resume normal
                }
            ]
        }
    ]
    
    for test in test_cases:
        print(f"\nTest: {test['name']}")
        controller = MockController()
        last_detection = time.time()
        is_turning = False
        
        for i, frame in enumerate(test['sequence']):
            print(f"\nFrame {i + 1}:")
            
            if 'is_turning' in frame:
                is_turning = frame['is_turning']
            
            last_detection, is_turning = handle_no_detection(
                controller, frame['detections'], last_detection, is_turning)
            
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
            
            print("✓ Frame test passed")
        
        print("\n✓ Scenario test passed")

if __name__ == "__main__":
    test_no_detection_behavior()
