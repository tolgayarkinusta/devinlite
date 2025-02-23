import numpy as np
import cv2
import supervision as sv
from dataclasses import dataclass
from typing import List, Tuple
import sys
import os
import math

# Import only the geometric calculation functions we need for testing
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from main import calculate_triangle_metrics, calculate_insole_length

@dataclass
class MockDetection:
    bbox: np.ndarray  # [x1, y1, x2, y2]
    class_id: int

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
        detections.append(MockDetection(bbox, 0))
        depth_map.set_depth(int(x), int(y), depth)
    
    # Add green buoys (class_id = 1)
    for x, y, depth in green_buoys:
        bbox = np.array([x-20, y-20, x+20, y+20])
        detections.append(MockDetection(bbox, 1))
        depth_map.set_depth(int(x), int(y), depth)
    
    # Add hazards (class_id = 2 or 3)
    for x, y, depth, class_id in hazards:
        bbox = np.array([x-20, y-20, x+20, y+20])
        detections.append(MockDetection(bbox, class_id))
        depth_map.set_depth(int(x), int(y), depth)
    
    return detections, depth_map.data

# Test scenarios
def test_basic_navigation():
    """Test basic navigation between red and green buoys"""
    frame_width, frame_height = 1280, 720
    
    # Red buoys on left, green on right
    red_buoys = [(400, 360, 5.0)]   # x, y, depth
    green_buoys = [(880, 360, 5.0)]
    hazards = []
    
    detections, depth_map = create_test_scenario(
        frame_width, frame_height, red_buoys, green_buoys, hazards)
    
    # Test triangle geometry calculations
    vertex_angle, base_width = calculate_triangle_metrics(
        400, 880,  # red and green x coordinates
        5.0, 5.0,  # depths
        frame_width
    )
    
    # Calculate insole length
    insole_length = calculate_insole_length(vertex_angle, base_width, 5.0)
    
    print("Basic Navigation Test:")
    print(f"Vertex Angle: {vertex_angle:.1f}°")
    print(f"Base Width: {base_width:.1f}m")
    print(f"Insole Length: {insole_length:.1f}m")
    
    # Verify calculations
    expected_angle = abs(math.degrees(math.atan2(480, 5000)))  # Expected angle from center
    print(f"Expected Vertex Angle: {expected_angle*2:.1f}°")  # *2 because we have symmetrical angles

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
    
    print("\nMultiple Obstacle Tests:")
    for red_x, red_depth, green_x, green_depth, desc in test_cases:
        vertex_angle, base_width = calculate_triangle_metrics(
            red_x, green_x,
            red_depth, green_depth,
            frame_width
        )
        avg_depth = (red_depth + green_depth) / 2
        insole_length = calculate_insole_length(vertex_angle, base_width, avg_depth)
        
        print(f"\n{desc}:")
        print(f"Vertex Angle: {vertex_angle:.1f}°")
        print(f"Base Width: {base_width:.1f}m")
        print(f"Insole Length: {insole_length:.1f}m")

if __name__ == "__main__":
    test_basic_navigation()
    test_multiple_obstacles()
