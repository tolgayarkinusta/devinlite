import math

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
    center_x = frame_width / 2
    angle_per_pixel = horizontal_fov / frame_width
    
    angle1 = (point1 - center_x) * angle_per_pixel
    angle2 = (point2 - center_x) * angle_per_pixel
    
    vertex_angle = abs(angle1 - angle2)
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
        insole_length: height of the triangle in meters
    """
    vertex_angle_rad = math.radians(vertex_angle)
    insole_length = base_width * math.sin(vertex_angle_rad / 2)
    return insole_length
