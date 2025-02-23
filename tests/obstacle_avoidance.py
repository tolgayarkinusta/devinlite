# Mock ZED camera for testing
class sl:
    class Mat:
        def __init__(self):
            self.data = None
        def get_data(self):
            return self.data
    
    class RuntimeParameters:
        pass
    
    class VIEW:
        LEFT = 0
    
    class MEASURE:
        DEPTH = 0
    
    class ERROR_CODE:
        SUCCESS = 0
import cv2
import numpy as np
import supervision as sv

class MockZED:
    def __init__(self):
        self.image = sl.Mat()
        self.depth = sl.Mat()
        self.runtime = sl.RuntimeParameters()
    
    def grab(self, params):
        return sl.ERROR_CODE.SUCCESS
    
    def retrieve_image(self, image, view):
        image.data = self.image.data.copy()
    
    def retrieve_measure(self, depth, measure):
        depth.data = self.depth.data.copy()

def calculate_triangle_metrics(x1, x2, d1, d2, frame_width):
    """Calculate vertex angle and base width for navigation triangle"""
    # Convert pixel coordinates to meters (assuming 90-degree FOV)
    x1_m = (x1 - frame_width/2) * d1 / (frame_width/2)
    x2_m = (x2 - frame_width/2) * d2 / (frame_width/2)
    
    # Calculate angles from center
    angle1 = abs(np.arctan2(x1_m, d1))
    angle2 = abs(np.arctan2(x2_m, d2))
    
    # Total vertex angle
    vertex_angle = np.degrees(angle1 + angle2)
    
    # Calculate base width using law of cosines
    base_width = np.sqrt(d1**2 + d2**2 - 2*d1*d2*np.cos(angle1 + angle2))
    
    return vertex_angle, base_width

def get_depth_at_point(depth_map, x, y):
    """Get depth value at a specific point, handling edge cases"""
    if depth_map is None:
        return None
    return float(depth_map[int(y), int(x)])

def find_optimal_path(detections, depth_map, frame_width):
    """Find optimal navigation path between buoys"""
    if not detections or len(detections) == 0:
        return None, None
    
    # Extract buoy positions and depths
    buoys = []
    for detection in detections:
        # Handle both tuple and Detection object formats
        if isinstance(detection, tuple):
            bbox = detection[0]
            class_id = detection[1]
        else:
            bbox = detection.xyxy[0]
            class_id = detection.class_id
        
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        depth = get_depth_at_point(depth_map, center_x, center_y)
        if depth is not None:
            buoys.append((center_x, depth, class_id))
    
    if len(buoys) < 2:
        # Single buoy case
        if len(buoys) == 1:
            return buoys[0][0], buoys[0][1]
        return None, None
    
    # Sort buoys by x-coordinate
    buoys.sort(key=lambda x: x[0])
    
    # Find red-green buoy pairs
    valid_pairs = []
    for i in range(len(buoys)-1):
        for j in range(i+1, len(buoys)):
            if buoys[i][2] == 0 and buoys[j][2] == 1:  # Red then Green
                valid_pairs.append((buoys[i], buoys[j]))
    
    if not valid_pairs:
        return None, None
    
    # Find pair with best angle and separation
    best_pair = None
    best_score = float('-inf')
    for red, green in valid_pairs:
        vertex_angle, base_width = calculate_triangle_metrics(
            red[0], green[0], red[1], green[1], frame_width)
        
        # Score based on angle and separation
        angle_score = -abs(vertex_angle - 45)  # Prefer 45-degree angles
        separation_score = base_width  # Prefer wider separation
        total_score = angle_score + separation_score
        
        if total_score > best_score:
            best_score = total_score
            best_pair = (red, green)
    
    if best_pair:
        target_x = (best_pair[0][0] + best_pair[1][0]) / 2
        target_depth = (best_pair[0][1] + best_pair[1][1]) / 2
        return target_x, target_depth
    
    return None, None

def calculate_insole_length(vertex_angle, base_width, height):
    """Calculate the length of the insole (height) of the navigation triangle"""
    if vertex_angle >= 180:
        return 0
    half_angle = np.radians(vertex_angle / 2)
    return base_width / (2 * np.tan(half_angle))

def navigate_usv(zed, model, controller, test_mode=False):
    """Navigate USV using camera input and YOLO detections
    Args:
        zed: ZED camera instance
        model: YOLO model instance
        controller: USV controller instance
        test_mode: If True, run in test mode
    """
    # Motor control constants
    STOP_PWM = 1500
    REVERSE_PWM = 1450
    MIN_PWM = 1540
    STRAIGHT_PWM = 1560
    MAX_PWM = 1570
    
    # Initialize state variables
    runtime_params = sl.RuntimeParameters()
    image = sl.Mat()
    depth = sl.Mat()
    
    # Process frame
    if not zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
        return
    
    zed.retrieve_image(image, sl.VIEW.LEFT)
    zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
    
    frame = image.get_data()
    depth_map = depth.get_data()
    frame_width = frame.shape[1]
    
    # Run YOLO detection
    results = model(frame)
    if not results or len(results) == 0:
        return
    
    # Convert to supervision Detections format
    if isinstance(results[0], sv.Detections):
        detections = results[0]
    else:
        # Convert YOLO results to supervision format
        detections = sv.Detections.from_ultralytics(results[0])
    
    # Process detections
    if len(detections) > 0:
        # Check for hazard buoys first
        has_hazard = False
        for i in range(len(detections)):
            bbox = detections.xyxy[i]
            class_id = detections.class_id[i]
            
            if class_id in [2, 3]:  # Yellow or black hazard buoy
                controller.set_servo(5, REVERSE_PWM)  # Reverse right motor
                controller.set_servo(6, REVERSE_PWM)  # Reverse left motor
                if test_mode:
                    return
                has_hazard = True
                break
        
        if not has_hazard:
            # If no hazard buoys, continue with normal navigation
            target_x, target_depth = find_optimal_path(detections, depth_map, frame_width)
            
            if target_x is not None:
                error = target_x - frame_width / 2
                error_ratio = abs(error) / (frame_width / 4)
                
                if error > 0:  # Turn right
                    controller.set_servo(5, STOP_PWM)  # Stop right motor
                    left_pwm = STRAIGHT_PWM + int((MAX_PWM - STRAIGHT_PWM) * min(error_ratio, 1.0))
                    controller.set_servo(6, left_pwm)
                elif error < 0:  # Turn left
                    right_pwm = STRAIGHT_PWM + int((MAX_PWM - STRAIGHT_PWM) * min(error_ratio, 1.0))
                    controller.set_servo(5, right_pwm)
                    controller.set_servo(6, STOP_PWM)  # Stop left motor
                else:  # Go straight
                    controller.set_servo(5, STRAIGHT_PWM)
                    controller.set_servo(6, STRAIGHT_PWM)
    
    if test_mode:
        return
    
    # Only show visualization in non-test mode
    cv2.imshow("Navigation", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return
