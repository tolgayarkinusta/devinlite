import math
import numpy as np

def calculate_triangle_metrics(point1, point2, depth1, depth2, frame_width, horizontal_fov=90):
    """
    Calculate triangle metrics with USV as vertex and two points forming base
    Args:
        point1, point2: x-coordinates in pixels
        depth1, depth2: depth values in meters
        frame_width: width of frame
        horizontal_fov: camera's horizontal field of view in degrees
    Returns:
        vertex_angle: angle between the two points from USV perspective
        base_width: actual distance between the two points
    """
    # Calculate angles from center for both points
    center_x = frame_width / 2
    angle_per_pixel = horizontal_fov / frame_width
    
    angle1 = (point1 - center_x) * angle_per_pixel
    angle2 = (point2 - center_x) * angle_per_pixel
    
    # Convert to radians
    angle1_rad = math.radians(angle1)
    angle2_rad = math.radians(angle2)
    
    # Calculate vertex angle (angle between the two points from USV perspective)
    vertex_angle = abs(angle1 - angle2)
    
    # Calculate base width using law of cosines
    # c² = a² + b² - 2ab*cos(C)
    base_width = math.sqrt(depth1**2 + depth2**2 - 
                          2 * depth1 * depth2 * math.cos(math.radians(vertex_angle)))
    
    return vertex_angle, base_width

def calculate_insole_length(vertex_angle, base_width, depth):
    """
    Calculate the height (insole) of the triangle from vertex to base
    Args:
        vertex_angle: angle at vertex in degrees
        base_width: width of the triangle base in meters
        depth: depth to the base in meters
    Returns:
        insole_length: height of the triangle
    """
    vertex_angle_rad = math.radians(vertex_angle)
    # Using the sine formula: height = base * sin(vertex_angle/2)
    insole_length = base_width * math.sin(vertex_angle_rad / 2)
    return insole_length

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
