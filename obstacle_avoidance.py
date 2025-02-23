import pyzed.sl as sl
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import math
import time

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

def get_depth_at_point(point_2d, depth_map):
    x, y = int(point_2d[0]), int(point_2d[1])
    if x >= 0 and x < depth_map.shape[1] and y >= 0 and y < depth_map.shape[0]:
        return depth_map[y, x]
    return None

def find_optimal_path(detections, depth_map, frame_width):
    """
    Find optimal path using triangle geometry with USV as vertex
    """
    # Find all buoys with their positions and depths
    red_buoys = []    # Left side
    green_buoys = []  # Right side
    hazard_buoys = [] # Yellow and black buoys
    
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox.xyxy[0]
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        depth = get_depth_at_point((center_x, center_y), depth_map)
        
        if depth is not None:
            if detection.class_id == 0:  # Red buoy
                red_buoys.append((center_x, depth))
            elif detection.class_id == 1:  # Green buoy
                green_buoys.append((center_x, depth))
            elif detection.class_id in [2, 3]:  # Hazard buoys (yellow=2 or black=3)
                hazard_buoys.append((center_x, depth, detection.class_id))
    
    # Handle single buoy detection cases
    if not red_buoys and not green_buoys:
        return None, None  # No buoys detected
    elif not red_buoys:  # Only green buoy detected
        closest_green = min(green_buoys, key=lambda x: x[1])
        return closest_green[0], closest_green[1]  # Turn left
    elif not green_buoys:  # Only red buoy detected
        closest_red = min(red_buoys, key=lambda x: x[1])
        return closest_red[0], closest_red[1]  # Turn right
    
    # Find closest red and green buoys
    closest_red = min(red_buoys, key=lambda x: x[1])
    closest_green = min(green_buoys, key=lambda x: x[1])
    
    # Calculate triangle metrics between red and green buoys
    gate_angle, gate_width = calculate_triangle_metrics(
        closest_red[0], closest_green[0],
        closest_red[1], closest_green[1],
        frame_width
    )
    
    # Find hazards between the buoys
    path_hazards = []
    for hx, hdepth in hazard_buoys:
        # Calculate angles to both gate buoys
        h_red_angle, h_red_dist = calculate_triangle_metrics(
            hx, closest_red[0], hdepth, closest_red[1], frame_width)
        h_green_angle, h_green_dist = calculate_triangle_metrics(
            hx, closest_green[0], hdepth, closest_green[1], frame_width)
        
        # If hazard is between the buoys
        if h_red_angle + h_green_angle <= gate_angle:
            path_hazards.append((hx, hdepth, h_red_angle/gate_angle))
    
    # Calculate target point
    if path_hazards:
        # Sort hazards by depth
        path_hazards.sort(key=lambda x: x[1])
        closest_hazard = path_hazards[0]
        
        # Position ratio (0 = near red buoy, 1 = near green buoy)
        hazard_ratio = closest_hazard[2]
        
        if hazard_ratio < 0.5:  # Hazard is closer to red buoy
            # Aim for 70% of the way to the green buoy
            target_x = closest_red[0] + (closest_green[0] - closest_red[0]) * 0.7
            target_depth = closest_hazard[1]
        else:  # Hazard is closer to green buoy
            # Aim for 30% of the way from red buoy
            target_x = closest_red[0] + (closest_green[0] - closest_red[0]) * 0.3
            target_depth = closest_hazard[1]
    else:
        # No hazards, aim for center
        target_x = (closest_red[0] + closest_green[0]) / 2
        target_depth = (closest_red[1] + closest_green[1]) / 2
    
    return target_x, target_depth

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

def navigate_usv(zed, model, controller):
    # Global variables for timing and state
    global hazard_detection_time, is_reversing, last_detection_time, is_turning
    if 'hazard_detection_time' not in globals():
        hazard_detection_time = None
    if 'is_reversing' not in globals():
        is_reversing = False
    if 'last_detection_time' not in globals():
        last_detection_time = time.time()
    if 'is_turning' not in globals():
        is_turning = False
    
    runtime_params = sl.RuntimeParameters()
    image = sl.Mat()
    depth = sl.Mat()
    
    while True:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            
            frame = image.get_data()
            depth_map = depth.get_data()
            frame_width = frame.shape[1]
            
            # Run YOLO detection
            results = model(frame)[0]
            detections = sv.Detections.from_yolov8(results)
            
            current_time = time.time()
            
            # Define motor control constants
            STOP_PWM = 1500
            REVERSE_PWM = 1450
            MIN_PWM = 1540
            STRAIGHT_PWM = 1560
            MAX_PWM = 1570

            # Reset detection timer when buoys are seen
            if len(detections) > 0:
                last_detection_time = current_time
            
            # Check for no-detection timeout
            if current_time - last_detection_time > 2.5:
                if not is_turning:
                    # First reverse for 2 seconds
                    controller.set_servo(5, REVERSE_PWM)
                    controller.set_servo(6, REVERSE_PWM)
                    time.sleep(2.0)
                    is_turning = True
                    return
                else:
                    # After reverse, start/continue turning until proper orientation
                    controller.set_servo(5, MIN_PWM)  # Right motor forward
                    controller.set_servo(6, REVERSE_PWM)  # Left motor reverse
                    return
            
            # Process detections if any
            target_x = None
            target_depth = None
            if len(detections) > 0:
                target_x, target_depth = find_optimal_path(detections, depth_map, frame_width)
                
                if target_x is not None:
                    frame_center_x = frame_width // 2
                    error = target_x - frame_center_x
                    
                    # Get all buoys for navigation decisions
                    red_buoys = []
                    green_buoys = []
                    hazard_buoys = []  # Yellow and black buoys
                    for detection in detections:
                        x1, y1, x2, y2 = detection.bbox.xyxy[0]
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        depth_val = get_depth_at_point((center_x, center_y), depth_map)
                        
                        if depth_val is not None:
                            if detection.class_id == 0:  # Red buoy
                                red_buoys.append((center_x, depth_val))
                            elif detection.class_id == 1:  # Green buoy
                                green_buoys.append((center_x, depth_val))
                            elif detection.class_id in [2, 3]:  # Yellow (2) or black (3) buoy
                                hazard_buoys.append((center_x, depth_val))
                    
                    # Check orientation during turn
                    if is_turning:
                        if red_buoys and green_buoys:
                            closest_red = min(red_buoys, key=lambda x: x[1])
                            closest_green = min(green_buoys, key=lambda x: x[1])
                            
                            # Only stop turning if green buoys are on the right
                            if closest_green[0] > closest_red[0]:
                                is_turning = False
                            else:
                                # Continue turning
                                controller.set_servo(5, MIN_PWM)
                                controller.set_servo(6, REVERSE_PWM)
                                return
                        else:
                            # Keep turning if we don't see both buoys
                            controller.set_servo(5, MIN_PWM)
                            controller.set_servo(6, REVERSE_PWM)
                            return
                    
                    # Normal navigation
                    if red_buoys and green_buoys:
                        closest_red = min(red_buoys, key=lambda x: x[1])
                        closest_green = min(green_buoys, key=lambda x: x[1])
                        
                        # Calculate triangle metrics
                        vertex_angle, base_width = calculate_triangle_metrics(
                            closest_red[0], closest_green[0],
                            closest_red[1], closest_green[1],
                            frame_width
                        )
                        
                        # Calculate insole length (height of triangle)
                        avg_depth = (closest_red[1] + closest_green[1]) / 2
                        insole_length = calculate_insole_length(vertex_angle, base_width, avg_depth)
                        
                        # Use the already defined motor control constants from above
                        
                        # Check for hazard buoys first
                        if hazard_buoys:
                            current_time = time.time()
                            
                            # Start reversing if not already
                            if not is_reversing:
                                hazard_detection_time = current_time
                                is_reversing = True
                            
                            # Continue reversing if within 2 seconds of last detection
                            if current_time - hazard_detection_time < 2.0:
                                controller.set_servo(5, REVERSE_PWM)
                                controller.set_servo(6, REVERSE_PWM)
                                return
                            else:
                                is_reversing = False
                                hazard_detection_time = None
                        
                        # Handle single buoy cases
                        if not red_buoys and green_buoys:  # Only green buoy
                            # Turn left - stop left motor, right motor at max
                            controller.set_servo(5, MAX_PWM)   # Right motor full
                            controller.set_servo(6, STOP_PWM)  # Left motor stopped
                            return
                        elif red_buoys and not green_buoys:  # Only red buoy
                            # Turn right - stop right motor, left motor at max
                            controller.set_servo(5, STOP_PWM)  # Right motor stopped
                            controller.set_servo(6, MAX_PWM)   # Left motor full
                            return
                        else:  # Both buoys detected - normal navigation
                            # Scale turn speed based on error magnitude
                            error_ratio = abs(error) / (frame_width / 4)  # Normalize to quarter of frame width
                            
                            if error > 0:  # Need to turn right
                                controller.set_servo(5, STOP_PWM)  # Stop right motor
                                # Scale left motor between MIN_PWM and MAX_PWM
                                left_pwm = MIN_PWM + (MAX_PWM - MIN_PWM) * min(error_ratio, 1.0)
                                controller.set_servo(6, int(left_pwm))
                            elif error < 0:  # Need to turn left
                                # Scale right motor between MIN_PWM and MAX_PWM
                                right_pwm = MIN_PWM + (MAX_PWM - MIN_PWM) * min(error_ratio, 1.0)
                                controller.set_servo(5, int(right_pwm))
                                controller.set_servo(6, STOP_PWM)  # Stop left motor
                            else:  # Go straight
                                controller.set_servo(5, STRAIGHT_PWM)
                                controller.set_servo(6, STRAIGHT_PWM)
                        
                        # Visualize triangle metrics (optional)
                        cv2.putText(frame, f"Vertex Angle: {vertex_angle:.1f}", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame, f"Insole Length: {insole_length:.1f}m", (10, 60),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        # No valid triangle, stop motors
                        controller.set_servo(5, STOP_PWM)
                        controller.set_servo(6, STOP_PWM)
                else:
                    # No valid path found, stop motors
                    controller.set_servo(5, STOP_PWM)
                    controller.set_servo(6, STOP_PWM)
            else:
                # No detections, stop motors
                controller.set_servo(5, STOP_PWM)
                controller.set_servo(6, STOP_PWM)
            
            # Display frame with visualizations
            cv2.imshow("Navigation", frame)
            
            # Break loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Stop motors before exiting
    controller.set_servo(5, STOP_PWM)
    controller.set_servo(6, STOP_PWM)
    cv2.destroyAllWindows()
