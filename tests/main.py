import pyzed.sl as sl
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import math
import time

# Motor control constants
STOP_PWM = 1500        # PWM value to stop motor
REVERSE_PWM = 1450     # PWM value for reverse
MIN_PWM = 1540         # Minimum PWM for normal operation
STRAIGHT_PWM = 1560    # PWM for straight movement
MAX_PWM = 1570         # Maximum PWM value

class USVController:
    def __init__(self, port="COM10", baud=115200):
        from MainSystemTest import USVController as BaseController
        self.controller = BaseController(port, baud)
        self.right_motor = 5  # Right motor pin
        self.left_motor = 6   # Left motor pin
        
        # State variables
        self.hazard_detection_time = None
        self.is_reversing = False
        self.last_detection_time = time.time()
        self.is_turning = False
    
    def set_servo(self, pin, pwm):
        """Control motor PWM values"""
        self.controller.set_servo(pin, pwm)
    
    def initialize(self):
        """Initialize the USV"""
        print("Arming vehicle...")
        self.controller.arm_vehicle()
        print("Vehicle armed!")
        print("Setting mode...")
        self.controller.set_mode("MANUAL")
        print("Mode set!")

def calculate_triangle_metrics(point1, point2, depth1, depth2, frame_width, horizontal_fov=90):
    """Calculate triangle metrics with USV as vertex"""
    center_x = frame_width / 2
    angle_per_pixel = horizontal_fov / frame_width
    
    angle1 = (point1 - center_x) * angle_per_pixel
    angle2 = (point2 - center_x) * angle_per_pixel
    
    vertex_angle = abs(angle1 - angle2)
    base_width = math.sqrt(depth1**2 + depth2**2 - 
                          2 * depth1 * depth2 * math.cos(math.radians(vertex_angle)))
    
    return vertex_angle, base_width

def get_depth_at_point(point_2d, depth_map):
    """Get depth value at given point"""
    x, y = int(point_2d[0]), int(point_2d[1])
    if x >= 0 and x < depth_map.shape[1] and y >= 0 and y < depth_map.shape[0]:
        return depth_map[y, x]
    return None

def process_detections(detections, depth_map, frame_width):
    """Process detections and find optimal path"""
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
            elif detection.class_id in [2, 3]:  # Hazard buoys
                hazard_buoys.append((center_x, depth))
    
    return red_buoys, green_buoys, hazard_buoys

def main():
    # Initialize ZED camera with recommended settings
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
    init_params.depth_minimum_distance = 0.20
    init_params.depth_maximum_distance = 40
    init_params.camera_disable_self_calib = True
    init_params.depth_stabilization = 80
    init_params.sensors_required = False
    init_params.enable_image_enhancement = True
    
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED camera")
        return
    
    # Initialize YOLO model
    model = YOLO("balonLarge54.pt")
    
    # Initialize USV controller
    controller = USVController()
    controller.initialize()
    
    runtime_params = sl.RuntimeParameters()
    image = sl.Mat()
    depth = sl.Mat()
    
    while True:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # Get camera data
            zed.retrieve_image(image, sl.VIEW.LEFT)
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            frame = image.get_data()
            depth_map = depth.get_data()
            frame_width = frame.shape[1]
            
            # Run detection
            results = model(frame)[0]
            detections = sv.Detections.from_yolov8(results)
            
            current_time = time.time()
            
            # Handle no detection timeout
            if len(detections) == 0:
                if current_time - controller.last_detection_time > 2.5:
                    if not controller.is_turning:
                        # Reverse for 2 seconds
                        controller.set_servo(controller.right_motor, REVERSE_PWM)
                        controller.set_servo(controller.left_motor, REVERSE_PWM)
                        time.sleep(2.0)
                        controller.is_turning = True
                    else:
                        # Turn until proper orientation
                        controller.set_servo(controller.right_motor, MIN_PWM)
                        controller.set_servo(controller.left_motor, REVERSE_PWM)
                    continue
            else:
                controller.last_detection_time = current_time
            
            # Process detections
            red_buoys, green_buoys, hazard_buoys = process_detections(detections, depth_map, frame_width)
            
            # Handle hazard detection (yellow or black buoys)
            if hazard_buoys:
                if not controller.is_reversing:
                    controller.hazard_detection_time = current_time
                    controller.is_reversing = True
                
                # Continue reversing for 2 more seconds after seeing other buoys
                if len(red_buoys) > 0 or len(green_buoys) > 0:
                    if current_time - controller.hazard_detection_time < 2.0:
                        controller.set_servo(controller.right_motor, REVERSE_PWM)  # 1450 PWM
                        controller.set_servo(controller.left_motor, REVERSE_PWM)   # 1450 PWM
                        continue
                    else:
                        controller.is_reversing = False
                        controller.hazard_detection_time = None
                else:
                    controller.set_servo(controller.right_motor, REVERSE_PWM)  # 1450 PWM
                    controller.set_servo(controller.left_motor, REVERSE_PWM)   # 1450 PWM
                    continue
            
            # Handle orientation verification during turn
            if controller.is_turning:
                if red_buoys and green_buoys:
                    closest_red = min(red_buoys, key=lambda x: x[1])
                    closest_green = min(green_buoys, key=lambda x: x[1])
                    
                    if closest_green[0] > closest_red[0]:  # Correct orientation
                        controller.is_turning = False
                    else:  # Continue turning
                        controller.set_servo(controller.right_motor, MIN_PWM)
                        controller.set_servo(controller.left_motor, REVERSE_PWM)
                        continue
            
            # Normal navigation
            if red_buoys and green_buoys:
                closest_red = min(red_buoys, key=lambda x: x[1])
                closest_green = min(green_buoys, key=lambda x: x[1])
                target_x = (closest_red[0] + closest_green[0]) / 2
                
                # Calculate error for proportional control
                error = target_x - frame_width / 2
                error_ratio = abs(error) / (frame_width / 4)
                
                if error > 0:  # Turn right
                    # Stop right motor, gradually increase left up to 1570
                    controller.set_servo(controller.right_motor, STOP_PWM)  # 1500 PWM
                    left_pwm = STRAIGHT_PWM + int((MAX_PWM - STRAIGHT_PWM) * min(error_ratio, 1.0))
                    controller.set_servo(controller.left_motor, left_pwm)  # 1560-1570 PWM
                elif error < 0:  # Turn left
                    # Stop left motor, gradually increase right up to 1570
                    right_pwm = STRAIGHT_PWM + int((MAX_PWM - STRAIGHT_PWM) * min(error_ratio, 1.0))
                    controller.set_servo(controller.right_motor, right_pwm)  # 1560-1570 PWM
                    controller.set_servo(controller.left_motor, STOP_PWM)  # 1500 PWM
                else:  # Go straight at 1560 PWM
                    controller.set_servo(controller.right_motor, STRAIGHT_PWM)  # 1560 PWM
                    controller.set_servo(controller.left_motor, STRAIGHT_PWM)  # 1560 PWM
            elif green_buoys:  # Only green buoy - turn left slowly
                controller.set_servo(controller.right_motor, STRAIGHT_PWM)  # 1560 PWM
                controller.set_servo(controller.left_motor, STOP_PWM)       # 1500 PWM
            elif red_buoys:  # Only red buoy - turn right slowly
                controller.set_servo(controller.right_motor, STOP_PWM)      # 1500 PWM
                controller.set_servo(controller.left_motor, STRAIGHT_PWM)   # 1560 PWM
            else:  # No valid navigation buoys
                controller.set_servo(controller.right_motor, STOP_PWM)
                controller.set_servo(controller.left_motor, STOP_PWM)
            
            # Display frame
            cv2.imshow("Navigation", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Cleanup
    controller.set_servo(controller.right_motor, STOP_PWM)
    controller.set_servo(controller.left_motor, STOP_PWM)
    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
