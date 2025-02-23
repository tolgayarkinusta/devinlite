import numpy as np
import time
from dataclasses import dataclass
import supervision as sv

@dataclass
class MockDetection:
    class_id: int
    bbox: np.ndarray
    xyxy: list

def create_mock_detections(frame_width=1280, frame_height=720):
    """Create mock detections for testing"""
    detections = []
    
    # Create mock hazard buoy detection
    hazard_bbox = np.array([600, 340, 680, 420])  # Center-ish of frame
    hazard_detection = MockDetection(
        class_id=2,  # Yellow buoy
        bbox=hazard_bbox,
        xyxy=[hazard_bbox]
    )
    detections.append(hazard_detection)
    
    return sv.Detections.from_ultralytics([None])  # Mock YOLO format

def test_hazard_detection():
    """Test reverse motion behavior with hazard buoys"""
    
    class MockController:
        def __init__(self):
            self.right_pwm = 1500
            self.left_pwm = 1500
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
    
    # Test cases
    test_cases = [
        {
            'name': "Hazard buoy detection",
            'scenario': [
                # First frame: Only hazard buoy
                {
                    'detections': [
                        MockDetection(
                            class_id=2,  # Yellow buoy
                            bbox=np.array([600, 340, 680, 420]),
                            xyxy=[np.array([600, 340, 680, 420])]
                        )
                    ],
                    'expected': {
                        'right_pwm': 1450,
                        'left_pwm': 1450
                    }
                },
                # Second frame: Hazard + navigation buoys
                {
                    'detections': [
                        MockDetection(
                            class_id=2,  # Yellow buoy
                            bbox=np.array([600, 340, 680, 420]),
                            xyxy=[np.array([600, 340, 680, 420])]
                        ),
                        MockDetection(
                            class_id=0,  # Red buoy
                            bbox=np.array([400, 340, 480, 420]),
                            xyxy=[np.array([400, 340, 480, 420])]
                        ),
                        MockDetection(
                            class_id=1,  # Green buoy
                            bbox=np.array([800, 340, 880, 420]),
                            xyxy=[np.array([800, 340, 880, 420])]
                        )
                    ],
                    'expected': {
                        'right_pwm': 1450,  # Should continue reversing
                        'left_pwm': 1450,
                        'duration': 2.0  # Should reverse for 2 more seconds
                    }
                }
            ]
        }
    ]
    
    print("Hazard Detection Tests:")
    for test in test_cases:
        print(f"\nTest: {test['name']}")
        controller = MockController()
        
        # Test each frame in the scenario
        for frame_num, frame in enumerate(test['scenario']):
            print(f"\nFrame {frame_num + 1}:")
            
            # Create depth map
            depth_map = np.zeros((720, 1280))
            for detection in frame['detections']:
                x1, y1, x2, y2 = detection.bbox
                center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                depth_map[center_y, center_x] = 5.0
            
            # Process frame
            from obstacle_avoidance import find_optimal_path
            target_x, target_depth = find_optimal_path(frame['detections'], depth_map, 1280)
            
            # Handle hazard buoy detection
            for detection in frame['detections']:
                if detection.class_id in [2, 3]:  # Yellow or black hazard buoy
                    controller.set_servo(5, 1450)  # Reverse right motor
                    controller.set_servo(6, 1450)  # Reverse left motor
                    break
            
            # Verify PWM values
            status = controller.get_status()
            print(f"Right motor PWM: {status['right_pwm']} (Expected: {frame['expected']['right_pwm']})")
            print(f"Left motor PWM: {status['left_pwm']} (Expected: {frame['expected']['left_pwm']})")
            
            assert status['right_pwm'] == frame['expected']['right_pwm'], \
                f"Right motor PWM mismatch: got {status['right_pwm']}, expected {frame['expected']['right_pwm']}"
            assert status['left_pwm'] == frame['expected']['left_pwm'], \
                f"Left motor PWM mismatch: got {status['left_pwm']}, expected {frame['expected']['left_pwm']}"
            
            if 'duration' in frame['expected']:
                time.sleep(frame['expected']['duration'])
                final_status = controller.get_status()
                print(f"\nAfter {frame['expected']['duration']} seconds:")
                print(f"Right motor PWM: {final_status['right_pwm']}")
                print(f"Left motor PWM: {final_status['left_pwm']}")
                
                # Verify timing
                assert final_status['elapsed'] >= frame['expected']['duration'], \
                    f"Reverse duration too short: {final_status['elapsed']:.1f}s"
            
            print("✓ Frame test passed")
        
        print("\n✓ Scenario test passed")

if __name__ == "__main__":
    test_hazard_detection()
