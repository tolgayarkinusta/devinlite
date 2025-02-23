import numpy as np
from dataclasses import dataclass

@dataclass
class MockDetection:
    class_id: int
    bbox: np.ndarray

def test_single_buoy_detection():
    """Test single buoy detection and motor control logic"""
    frame_width = 1280
    frame_height = 720
    
    class MockController:
        def __init__(self):
            self.right_pwm = 1500
            self.left_pwm = 1500
            self.log = []
        
        def set_servo(self, pin, pwm):
            if pin == 5:
                self.right_pwm = pwm
            elif pin == 6:
                self.left_pwm = pwm
            self.log.append(f"Pin {pin}: {pwm}")
    
    def handle_single_buoy(detection, controller):
        """Simplified single buoy handling logic"""
        if detection.class_id == 1:  # Green buoy - turn left
            controller.set_servo(5, 1570)  # Right motor full
            controller.set_servo(6, 1500)  # Left motor stopped
            return "LEFT"
        else:  # Red buoy - turn right
            controller.set_servo(5, 1500)  # Right motor stopped
            controller.set_servo(6, 1570)  # Left motor full
            return "RIGHT"
    
    # Test cases
    test_cases = [
        {
            'name': "Only green buoy",
            'detection': MockDetection(
                class_id=1,
                bbox=np.array([800, 360, 840, 400])
            ),
            'expected_turn': "LEFT",
            'expected_pwm': {'right': 1570, 'left': 1500}
        },
        {
            'name': "Only red buoy",
            'detection': MockDetection(
                class_id=0,
                bbox=np.array([400, 360, 440, 400])
            ),
            'expected_turn': "RIGHT",
            'expected_pwm': {'right': 1500, 'left': 1570}
        }
    ]
    
    # Run tests
    print("Single Buoy Detection Tests:")
    for test in test_cases:
        print(f"\nTest: {test['name']}")
        controller = MockController()
        
        # Test turn direction
        turn_direction = handle_single_buoy(test['detection'], controller)
        print(f"Turn direction: {turn_direction}")
        print(f"Expected direction: {test['expected_turn']}")
        
        # Verify PWM values
        print(f"Right motor PWM: {controller.right_pwm} (Expected: {test['expected_pwm']['right']})")
        print(f"Left motor PWM: {controller.left_pwm} (Expected: {test['expected_pwm']['left']})")
        
        # Assert correct behavior
        assert turn_direction == test['expected_turn'], \
            f"Wrong turn direction. Expected {test['expected_turn']}, got {turn_direction}"
        assert controller.right_pwm == test['expected_pwm']['right'], \
            f"Wrong right motor PWM. Expected {test['expected_pwm']['right']}, got {controller.right_pwm}"
        assert controller.left_pwm == test['expected_pwm']['left'], \
            f"Wrong left motor PWM. Expected {test['expected_pwm']['left']}, got {controller.left_pwm}"
        
        print("✓ Test passed")

if __name__ == "__main__":
    test_single_buoy_detection()
