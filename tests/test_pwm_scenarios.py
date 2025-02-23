import numpy as np
import time
from dataclasses import dataclass

# Constants from main.py
STOP_PWM = 1500        # PWM value to stop motor
REVERSE_PWM = 1450     # PWM value for reverse
MIN_PWM = 1540         # Minimum PWM for normal operation
STRAIGHT_PWM = 1560    # PWM for straight movement
MAX_PWM = 1570         # Maximum PWM value

@dataclass
class MockDetection:
    bbox: np.ndarray
    class_id: int
    xyxy: list

class MockController:
    def __init__(self):
        self.right_pwm = STOP_PWM
        self.left_pwm = STOP_PWM
        self.log = []
        self.start_time = time.time()
    
    def set_servo(self, pin, pwm):
        if pin == 5:  # Right motor
            self.right_pwm = pwm
        elif pin == 6:  # Left motor
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
            'elapsed': time.time() - self.start_time
        }

def test_single_buoy_detection():
    """Test single buoy detection scenarios"""
    controller = MockController()
    
    # Test green buoy only - should turn left
    detections = [
        MockDetection(
            bbox=np.array([800, 360, 840, 400]),
            class_id=1,  # Green buoy
            xyxy=[np.array([800, 360, 840, 400])]
        )
    ]
    
    # Verify PWM values for left turn
    controller.set_servo(5, STRAIGHT_PWM)  # Right motor 1560
    controller.set_servo(6, STOP_PWM)      # Left motor 1500
    status = controller.get_status()
    assert status['right_pwm'] == 1560, "Right motor PWM incorrect for left turn"
    assert status['left_pwm'] == 1500, "Left motor PWM incorrect for left turn"
    
    # Reset controller
    controller = MockController()
    
    # Test red buoy only - should turn right
    detections = [
        MockDetection(
            bbox=np.array([400, 360, 440, 400]),
            class_id=0,  # Red buoy
            xyxy=[np.array([400, 360, 440, 400])]
        )
    ]
    
    # Verify PWM values for right turn
    controller.set_servo(5, STOP_PWM)       # Right motor 1500
    controller.set_servo(6, STRAIGHT_PWM)   # Left motor 1560
    status = controller.get_status()
    assert status['right_pwm'] == 1500, "Right motor PWM incorrect for right turn"
    assert status['left_pwm'] == 1560, "Left motor PWM incorrect for right turn"

def test_hazard_detection():
    """Test hazard buoy detection behavior"""
    controller = MockController()
    
    # Test yellow hazard buoy
    detections = [
        MockDetection(
            bbox=np.array([640, 360, 680, 400]),
            class_id=2,  # Yellow buoy
            xyxy=[np.array([640, 360, 680, 400])]
        )
    ]
    
    # Should reverse (1450 PWM both motors)
    controller.set_servo(5, REVERSE_PWM)
    controller.set_servo(6, REVERSE_PWM)
    status = controller.get_status()
    assert status['right_pwm'] == 1450, "Right motor PWM incorrect for reverse"
    assert status['left_pwm'] == 1450, "Left motor PWM incorrect for reverse"
    
    # Add navigation buoy and verify 2-second continuation
    detections.append(
        MockDetection(
            bbox=np.array([400, 360, 440, 400]),
            class_id=0,  # Red buoy
            xyxy=[np.array([400, 360, 440, 400])]
        )
    )
    
    start_time = time.time()
    while time.time() - start_time < 2.0:
        assert controller.right_pwm == 1450, "Right motor should maintain reverse"
        assert controller.left_pwm == 1450, "Left motor should maintain reverse"
        time.sleep(0.1)

def test_no_detection():
    """Test behavior when no buoys are detected"""
    controller = MockController()
    start_time = time.time()
    
    # Wait for 2.5 seconds with no detections
    while time.time() - start_time < 2.5:
        time.sleep(0.1)
    
    # Should reverse for 2 seconds
    controller.set_servo(5, REVERSE_PWM)  # Right motor 1450
    controller.set_servo(6, REVERSE_PWM)  # Left motor 1450
    status = controller.get_status()
    assert status['right_pwm'] == 1450, "Right motor PWM incorrect for initial reverse"
    assert status['left_pwm'] == 1450, "Left motor PWM incorrect for initial reverse"
    
    time.sleep(2.0)
    
    # Should turn (right motor 1540, left motor 1450)
    controller.set_servo(5, MIN_PWM)      # Right motor 1540
    controller.set_servo(6, REVERSE_PWM)  # Left motor 1450
    status = controller.get_status()
    assert status['right_pwm'] == 1540, "Right motor PWM incorrect for turn"
    assert status['left_pwm'] == 1450, "Left motor PWM incorrect for turn"

if __name__ == "__main__":
    test_single_buoy_detection()
    test_hazard_detection()
    test_no_detection()
    print("All PWM scenario tests passed!")
