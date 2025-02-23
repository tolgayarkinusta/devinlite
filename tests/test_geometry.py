import math
import numpy as np
from geometry_utils import calculate_triangle_metrics, calculate_insole_length

def test_basic_geometry():
    """Test basic triangle geometry calculations"""
    frame_width = 1280
    
    # Test case 1: Symmetric case with equal depths
    print("Test Case 1: Symmetric buoys at equal depths")
    vertex_angle, base_width = calculate_triangle_metrics(
        400, 880,  # red and green x-coordinates
        5.0, 5.0,  # depths
        frame_width
    )
    insole_length = calculate_insole_length(vertex_angle, base_width, 5.0)
    print(f"Vertex Angle: {vertex_angle:.1f}°")
    print(f"Base Width: {base_width:.1f}m")
    print(f"Insole Length: {insole_length:.1f}m")
    
    # Test case 2: Different depths
    print("\nTest Case 2: Different depths")
    vertex_angle, base_width = calculate_triangle_metrics(
        400, 880,
        3.0, 6.0,  # red closer than green
        frame_width
    )
    avg_depth = (3.0 + 6.0) / 2
    insole_length = calculate_insole_length(vertex_angle, base_width, avg_depth)
    print(f"Vertex Angle: {vertex_angle:.1f}°")
    print(f"Base Width: {base_width:.1f}m")
    print(f"Insole Length: {insole_length:.1f}m")
    
    # Test case 3: Wide separation
    print("\nTest Case 3: Wide separation")
    vertex_angle, base_width = calculate_triangle_metrics(
        200, 1080,  # wider separation
        5.0, 5.0,
        frame_width
    )
    insole_length = calculate_insole_length(vertex_angle, base_width, 5.0)
    print(f"Vertex Angle: {vertex_angle:.1f}°")
    print(f"Base Width: {base_width:.1f}m")
    print(f"Insole Length: {insole_length:.1f}m")

if __name__ == "__main__":
    test_basic_geometry()
