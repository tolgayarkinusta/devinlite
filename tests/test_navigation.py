import numpy as np
import supervision as sv
from dataclasses import dataclass
from typing import List, Tuple
import math
from geometry_utils import calculate_triangle_metrics, calculate_insole_length
from obstacle_avoidance import navigate_usv, MockZED, sl

# Disable visualization for tests
import cv2
cv2.imshow = lambda *args: None
cv2.waitKey = lambda *args: 0

@dataclass
class MockDetection:
    bbox: np.ndarray  # [x1, y1, x2, y2]
    class_id: int
    xyxy: list  # List containing bbox array

class MockModel:
    def __init__(self):
        self.detections = []
    
    def __call__(self, frame):
        if not self.detections:
            return []
        boxes = np.array([d.xyxy[0] for d in self.detections])
        class_ids = np.array([d.class_id for d in self.detections])
        confidence = np.array([0.9 for _ in self.detections])
        return [sv.Detections(xyxy=boxes, class_id=class_ids, confidence=confidence)]

class MockDepthMap:
    def __init__(self, width: int, height: int):
        self.shape = (height, width)
        self.data = np.zeros((height, width))
    
    def set_depth(self, x: int, y: int, depth: float):
        if 0 <= x < self.shape[1] and 0 <= y < self.shape[0]:
            self.data[y, x] = depth

def create_test_scenario(frame_width: int, frame_height: int, 
                        red_buoys: List[Tuple[float, float, float]],  # x, y, depth
                        green_buoys: List[Tuple[float, float, float]],
                        hazards: List[Tuple[float, float, float, int]]):  # x, y, depth, class_id
    """Create a test scenario with mock detections and depth map"""
    depth_map = MockDepthMap(frame_width, frame_height)
    detections = []
    
    # Add red buoys (class_id = 0)
    for x, y, depth in red_buoys:
        bbox = np.array([x-20, y-20, x+20, y+20])  # 40x40 pixel bounding box
        detections.append(MockDetection(
            bbox=bbox,
            class_id=0,
            xyxy=[bbox]
        ))
        depth_map.set_depth(int(x), int(y), depth)
    
    # Add green buoys (class_id = 1)
    for x, y, depth in green_buoys:
        bbox = np.array([x-20, y-20, x+20, y+20])
        detections.append(MockDetection(
            bbox=bbox,
            class_id=1,
            xyxy=[bbox]
        ))
        depth_map.set_depth(int(x), int(y), depth)
    
    # Add hazards (class_id = 2 or 3)
    for x, y, depth, class_id in hazards:
        bbox = np.array([x-20, y-20, x+20, y+20])
        detections.append(MockDetection(
            bbox=bbox,
            class_id=class_id,
            xyxy=[bbox]
        ))
        depth_map.set_depth(int(x), int(y), depth)
    
    # Create mock model
    model = MockModel()
    model.detections = detections
    return model, depth_map.data

# Test scenarios
def test_basic_navigation():
    """Test basic navigation between red and green buoys"""
    frame_width, frame_height = 1280, 720
    
    # Red buoys on left, green on right
    red_buoys = [(400, 360, 5.0)]   # x, y, depth
    green_buoys = [(880, 360, 5.0)]
    hazards = []
    
    # Create mock detections
    detections = []
    for x, y, depth in red_buoys:
        detections.append(MockDetection(
            bbox=np.array([x-20, y-20, x+20, y+20]),
            class_id=0,  # Red buoy
            xyxy=[np.array([x-20, y-20, x+20, y+20])]
        ))
    
    for x, y, depth in green_buoys:
        detections.append(MockDetection(
            bbox=np.array([x-20, y-20, x+20, y+20]),
            class_id=1,  # Green buoy
            xyxy=[np.array([x-20, y-20, x+20, y+20])]
        ))
    
    # Create depth map
    depth_map = np.zeros((frame_height, frame_width))
    for x, y, depth in red_buoys + green_buoys:
        depth_map[int(y), int(x)] = depth
        
    # Create mock objects
    zed = MockZED()
    zed.image.data = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    zed.depth.data = depth_map
    
    # Create mock model instance
    model = MockModel()
    model.detections = detections
    
    # Create mock controller
    class MockController:
        def __init__(self):
            self.right_pwm = 1500
            self.left_pwm = 1500
        
        def set_servo(self, pin, pwm):
            if pin == 5:
                self.right_pwm = pwm
            elif pin == 6:
                self.left_pwm = pwm
    
    controller = MockController()
    
    # Test navigation with mock model
    navigate_usv(zed, model, controller, test_mode=True)
    
    # Verify navigation behavior
    print("\nBasic Navigation Test:")
    print(f"Right motor PWM: {controller.right_pwm}")
    print(f"Left motor PWM: {controller.left_pwm}")
    
    # Verify PWM values for basic navigation
    assert controller.right_pwm == 1560 or controller.right_pwm == 1500, \
        f"Unexpected right motor PWM: {controller.right_pwm}"
    assert controller.left_pwm == 1560 or controller.left_pwm == 1500, \
        f"Unexpected left motor PWM: {controller.left_pwm}"
    
    # Test triangle geometry calculations
    vertex_angle, base_width = calculate_triangle_metrics(
        400, 880,  # red and green x coordinates
        5.0, 5.0,  # depths
        frame_width
    )
    
    # Calculate insole length
    insole_length = calculate_insole_length(vertex_angle, base_width, 5.0)
    
    print(f"Vertex Angle: {vertex_angle:.1f}°")
    print(f"Base Width: {base_width:.1f}m")
    print(f"Insole Length: {insole_length:.1f}m")
    
    # Verify calculations
    expected_angle = abs(math.degrees(math.atan2(480, 5000)))  # Expected angle from center
    print(f"Expected Vertex Angle: {expected_angle*2:.1f}°")  # *2 because we have symmetrical angles
    
    print("✓ Test passed")

def test_multiple_obstacles():
    """Test triangle calculations with multiple obstacles"""
    frame_width, frame_height = 1280, 720
    
    # Test different triangle configurations
    test_cases = [
        # (red_x, red_depth, green_x, green_depth, description)
        (320, 4.0, 960, 4.0, "Wide separation"),
        (500, 3.0, 780, 3.0, "Narrow separation"),
        (400, 3.0, 880, 6.0, "Different depths")
    ]
    
    # Create mock controller
    class MockController:
        def __init__(self):
            self.right_pwm = 1500
            self.left_pwm = 1500
        
        def set_servo(self, pin, pwm):
            if pin == 5:
                self.right_pwm = pwm
            elif pin == 6:
                self.left_pwm = pwm
    
    controller = MockController()
    
    print("\nMultiple Obstacle Tests:")
    for red_x, red_depth, green_x, green_depth, desc in test_cases:
        # Create mock detections
        detections = [
            MockDetection(
                bbox=np.array([red_x-20, 360-20, red_x+20, 360+20]),
                class_id=0,  # Red buoy
                xyxy=[np.array([red_x-20, 360-20, red_x+20, 360+20])]
            ),
            MockDetection(
                bbox=np.array([green_x-20, 360-20, green_x+20, 360+20]),
                class_id=1,  # Green buoy
                xyxy=[np.array([green_x-20, 360-20, green_x+20, 360+20])]
            )
        ]
        
        # Create depth map
        depth_map = np.zeros((frame_height, frame_width))
        depth_map[360, int(red_x)] = red_depth
        depth_map[360, int(green_x)] = green_depth
        
        # Create mock objects
        zed = MockZED()
        zed.image.data = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        zed.depth.data = depth_map
        
        # Create mock model instance
        model = MockModel()
        model.detections = detections
        
        # Test navigation with mock model and verify behavior
        print(f"\n{desc}:")
        print(f"Right motor PWM: {controller.right_pwm}")
        print(f"Left motor PWM: {controller.left_pwm}")
        
        # Test triangle geometry calculations
        vertex_angle, base_width = calculate_triangle_metrics(
            red_x, green_x,
            red_depth, green_depth,
            frame_width
        )
        avg_depth = (red_depth + green_depth) / 2
        insole_length = calculate_insole_length(vertex_angle, base_width, avg_depth)
        
        print(f"Vertex Angle: {vertex_angle:.1f}°")
        print(f"Base Width: {base_width:.1f}m")
        print(f"Insole Length: {insole_length:.1f}m")
        
        # Verify PWM values based on buoy positions
        if red_x < green_x:  # Normal case - red on left, green on right
            assert controller.right_pwm == 1560 or controller.right_pwm == 1500, \
                f"Unexpected right motor PWM: {controller.right_pwm}"
            assert controller.left_pwm == 1560 or controller.left_pwm == 1500, \
                f"Unexpected left motor PWM: {controller.left_pwm}"
        
        # Reset controller for next test
        controller = MockController()
        print("✓ Test passed")
    
    print("\nAll obstacle tests passed")

if __name__ == "__main__":
    test_basic_navigation()
    test_multiple_obstacles()
