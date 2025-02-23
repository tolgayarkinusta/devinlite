import numpy as np
from dataclasses import dataclass

# Constants for PWM values
STOP_PWM = 1500
MIN_PWM = 1540
STRAIGHT_PWM = 1560
MAX_PWM = 1570

class MockController:
    def __init__(self):
        self.right_pwm = STOP_PWM
        self.left_pwm = STOP_PWM
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

def test_pwm_constraints():
    """Test motor control with new PWM constraints"""
    frame_width = 1280
    controller = MockController()
    
    test_cases = [
        {
            'name': "Straight movement",
            'error': 0,
            'expected': {
                'right_pwm': STRAIGHT_PWM,
                'left_pwm': STRAIGHT_PWM
            }
        },
        {
            'name': "Sharp right turn",
            'error': frame_width/4,  # Maximum error
            'expected': {
                'right_pwm': STOP_PWM,
                'left_pwm': MAX_PWM
            }
        },
        {
            'name': "Slight right turn",
            'error': frame_width/8,  # Half maximum error
            'expected': {
                'right_pwm': STOP_PWM,
                'left_pwm': MIN_PWM + (MAX_PWM - MIN_PWM) / 2
            }
        },
        {
            'name': "Sharp left turn",
            'error': -frame_width/4,  # Maximum error
            'expected': {
                'right_pwm': MAX_PWM,
                'left_pwm': STOP_PWM
            }
        },
        {
            'name': "Slight left turn",
            'error': -frame_width/8,  # Half maximum error
            'expected': {
                'right_pwm': MIN_PWM + (MAX_PWM - MIN_PWM) / 2,
                'left_pwm': STOP_PWM
            }
        }
    ]
    
    print("PWM Constraint Tests:")
    for test in test_cases:
        print(f"\nTest: {test['name']}")
        
        # Calculate error ratio
        error_ratio = abs(test['error']) / (frame_width / 4)
        
        if test['error'] > 0:  # Need to turn right
            controller.set_servo(5, STOP_PWM)  # Stop right motor
            # Scale left motor between MIN_PWM and MAX_PWM
            left_pwm = MIN_PWM + (MAX_PWM - MIN_PWM) * min(error_ratio, 1.0)
            controller.set_servo(6, int(left_pwm))
        elif test['error'] < 0:  # Need to turn left
            # Scale right motor between MIN_PWM and MAX_PWM
            right_pwm = MIN_PWM + (MAX_PWM - MIN_PWM) * min(error_ratio, 1.0)
            controller.set_servo(5, int(right_pwm))
            controller.set_servo(6, STOP_PWM)  # Stop left motor
        else:  # Go straight
            controller.set_servo(5, STRAIGHT_PWM)
            controller.set_servo(6, STRAIGHT_PWM)
        
        status = controller.get_status()
        print(f"Right motor PWM: {status['right_pwm']} (Expected: {test['expected']['right_pwm']})")
        print(f"Left motor PWM: {status['left_pwm']} (Expected: {test['expected']['left_pwm']})")
        
        # Verify PWM values
        assert abs(status['right_pwm'] - test['expected']['right_pwm']) <= 1, \
            f"Right motor PWM mismatch: got {status['right_pwm']}, expected {test['expected']['right_pwm']}"
        assert abs(status['left_pwm'] - test['expected']['left_pwm']) <= 1, \
            f"Left motor PWM mismatch: got {status['left_pwm']}, expected {test['expected']['left_pwm']}"
        
        # Verify PWM constraints
        if status['right_pwm'] != STOP_PWM:
            assert status['right_pwm'] >= MIN_PWM, f"Right motor PWM {status['right_pwm']} below minimum {MIN_PWM}"
        if status['left_pwm'] != STOP_PWM:
            assert status['left_pwm'] >= MIN_PWM, f"Left motor PWM {status['left_pwm']} below minimum {MIN_PWM}"
        
        print("✓ Test passed")
        
        # Reset controller for next test
        controller = MockController()

if __name__ == "__main__":
    test_pwm_constraints()
