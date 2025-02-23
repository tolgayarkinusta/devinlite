import math

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

def test_motor_control():
    """Test motor control based on geometric calculations"""
    controller = MockController()
    frame_width = 1280
    base_speed = 1550
    
    test_cases = [
        {
            'name': "Center path",
            'target_x': 640,  # Center of frame
            'vertex_angle': 30,
            'insole_length': 4.0,
            'expected': {'right_pwm': 1560, 'left_pwm': 1560}  # Straight ahead at 1560
        },
        {
            'name': "Slight right turn",
            'target_x': 740,  # Slightly right of center
            'vertex_angle': 45,
            'insole_length': 3.0,
            'expected': {'right_pwm': 1500, 'left_pwm': 1535}  # Right turn, gradual increase
        },
        {
            'name': "Sharp right turn",
            'target_x': 960,  # Far right
            'vertex_angle': 60,
            'insole_length': 3.0,
            'expected': {'right_pwm': 1500, 'left_pwm': 1570}  # Right turn, maximum speed
        },
        {
            'name': "Slight left turn",
            'target_x': 540,  # Slightly left of center
            'vertex_angle': 45,
            'insole_length': 3.0,
            'expected': {'right_pwm': 1535, 'left_pwm': 1500}  # Left turn, gradual increase
        },
        {
            'name': "Sharp left turn",
            'target_x': 320,  # Far left
            'vertex_angle': 60,
            'insole_length': 3.0,
            'expected': {'right_pwm': 1570, 'left_pwm': 1500}  # Left turn, maximum speed
        }
    ]
    
    print("Motor Control Tests:")
    for test in test_cases:
        print(f"\nTest: {test['name']}")
        
        # Calculate error and PWM values
        frame_center_x = frame_width // 2
        error = test['target_x'] - frame_center_x
        
        straight_speed = 1560
        turn_speed = 1570
        
        # Scale turn speed based on error magnitude
        error_ratio = abs(error) / (frame_width / 4)  # Normalize to quarter of frame width
        turn_pwm = 1500 + (turn_speed - 1500) * min(error_ratio, 1.0)
        
        # Apply motor control
        if error > 0:  # Need to turn right
            controller.set_servo(5, 1500)  # Stop right motor
            controller.set_servo(6, int(turn_pwm))  # Increase left motor gradually
        elif error < 0:  # Need to turn left
            controller.set_servo(5, int(turn_pwm))  # Increase right motor gradually
            controller.set_servo(6, 1500)  # Stop left motor
        else:  # Go straight
            controller.set_servo(5, straight_speed)
            controller.set_servo(6, straight_speed)
        
        # Get current status
        status = controller.get_status()
        print(f"Right motor PWM: {status['right_pwm']}")
        print(f"Left motor PWM: {status['left_pwm']}")
        print(f"Expected right PWM: {test['expected']['right_pwm']}")
        print(f"Expected left PWM: {test['expected']['left_pwm']}")
        
        # Reset controller for next test
        controller = MockController()

if __name__ == "__main__":
    test_motor_control()
