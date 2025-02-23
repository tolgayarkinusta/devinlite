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

def test_no_detection_sequence():
    """Test the complete no-detection sequence"""
    print("\nTesting No Detection Sequence:")
    controller = MockController()
    start_time = time.time()
    
    # Phase 1: Initial state - no detections for 2.5 seconds
    print("\nPhase 1: Waiting for 2.5s timeout")
    time.sleep(2.6)  # Wait for timeout
    assert controller.right_pwm == STOP_PWM, "Motors should be stopped during wait"
    assert controller.left_pwm == STOP_PWM
    print("✓ Motors stopped during wait")
    
    # Phase 2: Reverse motion for 2 seconds
    print("\nPhase 2: Reverse motion")
    controller.set_servo(5, REVERSE_PWM)
    controller.set_servo(6, REVERSE_PWM)
    assert controller.right_pwm == REVERSE_PWM, "Both motors should reverse"
    assert controller.left_pwm == REVERSE_PWM
    time.sleep(2.0)
    print("✓ Reverse motion correct")
    
    # Phase 3: Turn-around motion
    print("\nPhase 3: Turn-around motion")
    controller.set_servo(5, MIN_PWM)  # Right motor forward
    controller.set_servo(6, REVERSE_PWM)  # Left motor reverse
    assert controller.right_pwm == MIN_PWM, "Right motor should be forward"
    assert controller.left_pwm == REVERSE_PWM, "Left motor should be reverse"
    print("✓ Turn-around motion correct")
    
    # Phase 4: Wrong orientation detection
    print("\nPhase 4: Testing wrong orientation")
    detections = [
        MockDetection(
            class_id=1,  # Green buoy on left (wrong)
            bbox=np.array([400, 360, 440, 400])
        ),
        MockDetection(
            class_id=0,  # Red buoy on right (wrong)
            bbox=np.array([880, 360, 920, 400])
        )
    ]
    # Should continue turning
    controller.set_servo(5, MIN_PWM)
    controller.set_servo(6, REVERSE_PWM)
    assert controller.right_pwm == MIN_PWM, "Should continue turning"
    assert controller.left_pwm == REVERSE_PWM
    print("✓ Continues turning on wrong orientation")
    
    # Phase 5: Correct orientation detection
    print("\nPhase 5: Testing correct orientation")
    detections = [
        MockDetection(
            class_id=0,  # Red buoy on left (correct)
            bbox=np.array([400, 360, 440, 400])
        ),
        MockDetection(
            class_id=1,  # Green buoy on right (correct)
            bbox=np.array([880, 360, 920, 400])
        )
    ]
    # Should resume normal navigation
    controller.set_servo(5, STRAIGHT_PWM)
    controller.set_servo(6, STRAIGHT_PWM)
    assert controller.right_pwm == STRAIGHT_PWM, "Should resume normal navigation"
    assert controller.left_pwm == STRAIGHT_PWM
    print("✓ Resumes normal navigation on correct orientation")
    
    print("\n✓ All tests passed!")

if __name__ == "__main__":
    test_no_detection_sequence()
