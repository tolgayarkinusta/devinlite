import numpy as np
import time
import supervision as sv
from dataclasses import dataclass

@dataclass
class MockDetection:
    bbox: np.ndarray
    class_id: int
    
    @property
    def xyxy(self):
        return [self.bbox]

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
    import time
    start_time = time.time()
    # Import dependencies
    import sys
    from obstacle_avoidance import navigate_usv, MockZED, sl
    
    # Disable visualization for tests
    import cv2
    cv2.imshow = lambda *args: None
    cv2.waitKey = lambda *args: 0
    
    frame_width = 1280
    frame_height = 720
    
    # Test cases
    test_cases = [
        {
            'name': "Yellow hazard buoy only",
            'detections': [
                MockDetection(
                    bbox=np.array([640, 360, 680, 400]),
                    class_id=2  # Yellow buoy
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
                    class_id=3  # Black buoy
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
                    class_id=2  # Yellow buoy
                ),
                MockDetection(
                    bbox=np.array([400, 360, 440, 400]),
                    class_id=0  # Red buoy
                ),
                MockDetection(
                    bbox=np.array([880, 360, 920, 400]),
                    class_id=1  # Green buoy
                )
            ],
            'expected': {
                'action': "reverse",
                'right_pwm': 1450,
                'left_pwm': 1450
            }
        }
    ]
    
    # Create mock depth map
    depth_map = create_mock_depth_map(frame_width, frame_height)
    for test_case in test_cases:
        for detection in test_case['detections']:
            x1, y1, x2, y2 = detection.bbox
            center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
            depth_map[center_y, center_x] = 5.0
    
    # Create mock ZED camera
    class MockZED:
        def __init__(self):
            self.image = sl.Mat()
            self.depth = sl.Mat()
            self.image.data = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            self.depth.data = depth_map
            self.runtime = sl.RuntimeParameters()
        
        def grab(self, params):
            return sl.ERROR_CODE.SUCCESS
        
        def retrieve_image(self, image, view):
            image.data = self.image.data.copy()
        
        def retrieve_measure(self, depth, measure):
            depth.data = self.depth.data.copy()
    
    # Mock YOLO model
    class MockModel:
        def __init__(self):
            self.detections = []
            
        def __call__(self, frame):
            if not self.detections:
                return []
            
            # Convert mock detections to supervision Detections
            boxes = []
            class_ids = []
            for detection in self.detections:
                boxes.append(detection.xyxy[0])  # Use xyxy instead of bbox
                class_ids.append(detection.class_id)
            
            boxes = np.array(boxes)
            class_ids = np.array(class_ids)
            confidences = np.array([0.9] * len(boxes))
            
            return [sv.Detections(
                xyxy=boxes,
                class_id=class_ids,
                confidence=confidences
            )]
    
    # Create mock classes
    controller = MockController()
    
    # Test cases defined at the top of the file
    
    print("Reverse Motion Tests:")
    for test in test_cases:
        print(f"\nTest: {test['name']}")
        
        # Create mock ZED camera and model
        zed = MockZED()
        model = MockModel()
        model.detections = test['detections']  # Store detections for mock
        
        # Test navigation behavior with test mode
        navigate_usv(zed, model, controller, test_mode=True)
        
        status = controller.get_status()
        
        # Verify PWM values
        print(f"Right motor PWM: {status['right_pwm']} (Expected: {test['expected']['right_pwm']})")
        print(f"Left motor PWM: {status['left_pwm']} (Expected: {test['expected']['left_pwm']})")
        
        assert status['right_pwm'] == test['expected']['right_pwm'], \
            f"Right motor PWM mismatch: got {status['right_pwm']}, expected {test['expected']['right_pwm']}"
        assert status['left_pwm'] == test['expected']['left_pwm'], \
            f"Left motor PWM mismatch: got {status['left_pwm']}, expected {test['expected']['left_pwm']}"
        
        # Verify test completed within timeout
        assert time.time() - start_time < 5.0, "Test exceeded timeout"
        
        print("✓ Test passed")
        
        # Reset controller for next test
        controller = MockController()

if __name__ == "__main__":
    test_reverse_motion()
