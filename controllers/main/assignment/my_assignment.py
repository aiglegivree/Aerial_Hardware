import numpy as np
import time
import cv2
import os
import threading
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R
from scipy.interpolate import CubicSpline
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# The available ground truth state measurements can be accessed by calling sensor_data[item]. All values of "item" are provided as defined in main.py within the function read_sensors.
# The "item" values that you may later retrieve for the hardware project are:
# "x_global": Global X position
# "y_global": Global Y position
# "z_global": Global Z position
# 'v_x": Global X velocity
# "v_y": Global Y velocity
# "v_z": Global Z velocity
# "ax_global": Global X acceleration
# "ay_global": Global Y acceleration
# "az_global": Global Z acceleration (With gravtiational acceleration subtracted)
# "roll": Roll angle (rad)
# "pitch": Pitch angle (rad)
# "yaw": Yaw angle (rad)
# "q_x": X Quaternion value
# "q_y": Y Quaternion value
# "q_z": Z Quaternion value
# "q_w": W Quaternion value

# A link to further information on how to access the sensor data on the Crazyflie hardware for the hardware practical can be found here: https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/api/logs/#stateestimate
INIT = 0
HOVER = 1
DETECT_1 = 2
LATERAL_TRAVEL = 3
DETECT_2 = 4
COMPUTE_GATE_POS = 5
TRAVEL_GATE = 6
COMPUTE_PATH = 7
RACE = 8


MIN_HEIGHT_PIXELS = 6    # Au lieu de 20. On accepte une porte qui fait au moins 6 pixels de haut.
MIN_AREA_PIXELS = 15     # Au lieu de 75. On accepte un tout petit bloc de pixels roses.

class MyAssignment:
    def __init__(self):
        self.state = INIT
        self.start_x = 0
        self.start_y = 0
        self.start_yaw = 0
        self.gate_pos = np.zeros((5, 4)) 
        self.gate_corner = np.zeros((5, 4, 3)) 
        self.yaw_rate = np.pi/6
        self.req_frames = 5
        self.frames_detected = 0
        self.dynamic_z = 1.15

        # --- Variables de recherche ---
        self.yaw_accumulated = 0.0          
        self.is_vertical_search_active = False
        self.vertical_search_target = 2.0
        
        # --- NOUVEAU : Variables pour l'angle mort (Random Walk) ---
        self.is_repositioning = False
        self.reposition_target_x = 0.0
        self.reposition_target_y = 0.0

        # --- Variables d'optique et de mémoire ---
        self.current_estimated_dist = 0.0
        self.look_at_x = 0.0
        self.look_at_y = 0.0
        self.ready_to_measure = False

        self.P = None
        self.r = None
        self.r_corners = np.zeros((4, 3))
        self.Q = None
        self.s = None
        self.s_corners = np.zeros((4, 3))

        self.is_approching_gate = True
        self.approach_distance = 0.3 

        self.stop_x = None
        self.stop_y = None

        self.current_gate_corners_3d = None
        
        self.x_target = None
        self.y_target = None
        self.z_target = None
        self.yaw_target = None

        #for travel splines
        self.cs_x = None
        self.cs_y = None
        self.cs_z = None
        self.spline_t = 0.0

        self.app_x = 0.0
        self.app_y = 0.0
        self.mid_x = 0.0 # NOUVEAU
        self.mid_y = 0.0 # NOUVEAU
        self.exit_x = 0.0
        self.exit_y = 0.0
        self.gate_yaw_target = 0.0
        self.gate_z_target = 0.0
        
        self.travel_phase = 0 # NOUVEAU : 0=Approche, 1=Centre, 2=Sortie
        self.curr_gate_index = 0
        self.expected_cadrans = [4, 2, 0, 10, 8]

        self.est_gate_pos_2d = np.zeros((5, 2))
        self.est_gate_pos_2d_bis = np.zeros((5, 2))

        self.memorized_gate_width = None # NOUVEAU : Mémoire pour contrer l'occlusion

        self.racing_waypoints = []
        self.racing_waypoint_index = 0

        # Toggle to enable/disable final race plot generation
        self.show_plot = False
        self.show_terminal_prints = False

    def terminal_print(self, message):
        if self.show_terminal_prints:
            print(message)

    def get_corner_positions_2d_sorted(self, contour):
        #to get the four corners postions in 3D space

        perimeter = cv2.arcLength(contour, True)
        epsilon = 0.03 * perimeter 
        approx = cv2.approxPolyDP(contour, epsilon, True)

        corners_2d = None

        if len(approx) == 4:
            corners_2d = approx.reshape(4, 2)
        else:
            # PLAN B : Si le contour est bruité ou un peu arrondi, 
            # on l'enferme dans le plus petit rectangle orienté possible
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            corners_2d = np.int32(box)

        #top-bottom sorting
        sorted_indices = np.argsort(corners_2d[:, 1])
        top_corners = corners_2d[sorted_indices[:2]]    # Les 2 points les plus hauts
        bottom_corners = corners_2d[sorted_indices[2:]]

        #left-right sorting
        top_left, top_right = top_corners[np.argsort(top_corners[:, 0])]
        bottom_left, bottom_right = bottom_corners[np.argsort(bottom_corners[:, 0])]

        corners_2d_sorted = np.array([top_left, top_right, bottom_right, bottom_left])

        return corners_2d_sorted


    def compute_command(self, sensor_data, camera_data, dt):

        # NOTE: Displaying the camera image with cv2.imshow() will throw an error because GUI operations should be performed in the main thread.
        # If you want to display the camera image you can call it in main.py.
        
        # Take off example
        '''if sensor_data['z_global'] < 0.49:
            control_command = [sensor_data['x_global'], sensor_data['y_global'], 1.0, sensor_data['yaw']]
            return control_command'''

        # ---- YOUR CODE HERE ----

        if self.state == INIT:
            self.start_x = sensor_data['x_global']
            self.start_y = sensor_data['y_global']
            self.start_yaw = sensor_data['yaw']
            self.state = HOVER
            self.terminal_print("Initialized, transitioning to HOVER state")

        if self.state == HOVER:
            control_command = [self.start_x, self.start_y, 1.15, self.start_yaw]
            if abs(sensor_data['z_global'] - 1.15) < 0.05:
                self.state = DETECT_1
                self.yaw_target = sensor_data['yaw']
                self.terminal_print("Reached hover altitude, transitioning to DETECT_1 state")
        
        if self.state == DETECT_1:
            gate_valid = False
            gate_in_sight = False
            img_bgr = cv2.cvtColor(camera_data, cv2.COLOR_BGRA2BGR)
            img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            lower_pink = np.array([140,50,50])
            upper_pink = np.array([160,255,255])
            mask = cv2.inRange(img_hsv, lower_pink, upper_pink)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

            if len(contours) > 0:
                sorted_contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[3], reverse=True)
                
                H = mask.shape[0]
                W = mask.shape[1]
                focal_length = 161.013922282
                real_height = 0.4
                pitch_angle = abs(sensor_data['pitch'])
                valid_contour = None
                
                for c in sorted_contours:
                    x, y, w, h = cv2.boundingRect(c)
                    contour_area = cv2.contourArea(c)
                    
                    if contour_area <= MIN_AREA_PIXELS or h <= MIN_HEIGHT_PIXELS:
                        continue
                        
                    padding = 10
                    touches_top = (y < padding)
                    touches_bottom = (y + h > H - padding)
                    touches_left = (x < padding)
                    touches_right = (x + w > W - padding)
                    is_partially_occluded = touches_top or touches_bottom or touches_left or touches_right
                    
                    screen_area = W * H
                    coverage_ratio = contour_area / screen_area
                    is_dangerously_close = False
                    if is_partially_occluded and coverage_ratio > 0.60:
                        is_dangerously_close = True
                        is_partially_occluded = False
                        
                    if is_partially_occluded:
                        continue 
                        
                    h_corrige = h / max(np.cos(pitch_angle), 0.1)
                    estimated_distance = (focal_length * real_height) / h_corrige
                    
                    u_c = x + w/2.0
                    drone_x = sensor_data['x_global']
                    drone_y = sensor_data['y_global']
                    drone_yaw = sensor_data['yaw']

                    angle_offset = np.arctan2((u_c - W / 2.0), focal_length)
                    est_gate_yaw = drone_yaw - angle_offset
                    
                    est_gate_x = drone_x + estimated_distance * np.cos(est_gate_yaw)
                    est_gate_y = drone_y + estimated_distance * np.sin(est_gate_yaw)
                    
                    center_x, center_y = 4.0, 4.0
                    angle_rad = np.arctan2(est_gate_y - center_y, est_gate_x - center_x)
                    angle_deg = np.degrees(angle_rad)
                    
                    clock_angle = (360 - angle_deg + 15) % 360
                    expected_center_angle = self.expected_cadrans[self.curr_gate_index] * 30 + 15
                    
                    angle_diff = abs(clock_angle - expected_center_angle)
                    angle_diff = min(angle_diff, 360 - angle_diff)
                    
                    if angle_diff <= 45.0:
                        valid_contour = c
                        best_x, best_y, best_w, best_h = x, y, w, h
                        best_dist = estimated_distance
                        best_est_x, best_est_y = est_gate_x, est_gate_y
                        best_danger = is_dangerously_close
                        break 
                        
                if valid_contour is not None:
                    tallest_contour = valid_contour
                    x, y, w, h = best_x, best_y, best_w, best_h
                    estimated_distance = best_dist
                    est_gate_x, est_gate_y = best_est_x, best_est_y
                    is_dangerously_close = best_danger
                       
                    if is_dangerously_close:
                        self.terminal_print("DANGER PROXIMITÉ 1 : Traversée forcée !")
                        self.state = TRAVEL_GATE 
                        yaw = sensor_data['yaw']
                        self.exit_x = sensor_data['x_global'] + 1.5 * np.cos(yaw)
                        self.exit_y = sensor_data['y_global'] + 1.5 * np.sin(yaw)
                        self.gate_z_target = sensor_data['z_global']
                        self.gate_yaw_target = yaw
                        self.spline_t = 2.0 
                    else:
                        self.yaw_accumulated = 0.0
                        self.is_vertical_search_active = False
                        
                        self.est_gate_pos_2d[self.curr_gate_index, 0] = est_gate_x
                        self.est_gate_pos_2d[self.curr_gate_index, 1] = est_gate_y

                        gate_in_sight = True 
                        
                        self.look_at_x = est_gate_x
                        self.look_at_y = est_gate_y
                        self.current_estimated_dist = estimated_distance

                        v_c = y + h / 2.0  
                        center_y_img = H / 2.0 
                        if v_c < center_y_img - 20: 
                            self.dynamic_z = sensor_data['z_global'] + 0.1 
                        elif v_c > center_y_img + 20:
                            self.dynamic_z = sensor_data['z_global'] - 0.1
                        else:
                            self.dynamic_z = sensor_data['z_global']
                        
                        if self.stop_x is None: 
                            self.stop_x = sensor_data['x_global']
                            self.stop_y = sensor_data['y_global']
                            
                        global_speed = np.linalg.norm([sensor_data['v_x'], sensor_data['v_y'], sensor_data['v_z']])
                        if global_speed < 0.1: 
                            self.frames_detected += 1
                            if self.frames_detected >= self.req_frames:
                                corners_2d_sorted = self.get_corner_positions_2d_sorted(tallest_contour)

                                quaternion = [sensor_data['q_x'], sensor_data['q_y'], sensor_data['q_z'], sensor_data['q_w']]
                                R_body_to_world = R.from_quat(quaternion).as_matrix()
                                R_cam_to_body = np.array([[0,0,1],[-1,0,0],[0,-1,0]])

                                for i, corner in enumerate(corners_2d_sorted):
                                    v_camera = np.array([corner[0] - W/2, corner[1] - H/2, focal_length])
                                    self.r_corners[i] = R_body_to_world @ (R_cam_to_body @ v_camera)

                                pos_drone = np.array([sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global']])
                                self.P = pos_drone + (R_body_to_world @ np.array([0.03, 0, 0.01]))
                                gate_valid = True
                                
                                self.memorized_gate_width = (w * estimated_distance) / focal_length
                                self.frames_detected = 0 
                        else:
                            self.frames_detected = 0
                else:
                    self.frames_detected = 0
            else:
                self.frames_detected = 0

            # --- TRANSITIONS DETECT 1 ---
            if self.state == TRAVEL_GATE:
                control_command = [self.exit_x, self.exit_y, self.gate_z_target, self.gate_yaw_target]
                
            elif gate_valid:
                self.state = LATERAL_TRAVEL
                lateral_travel = min(1.0, max(0.35, self.current_estimated_dist * 0.25))
                self.terminal_print(f"Porte détectée à {self.current_estimated_dist:.2f}m. Déplacement latéral de {lateral_travel:.2f}m")
                
                yaw = sensor_data['yaw']
                self.x_target = sensor_data['x_global'] + lateral_travel * np.sin(yaw)
                self.y_target = sensor_data['y_global'] - lateral_travel * np.cos(yaw)
                self.z_target = sensor_data['z_global']
                self.yaw_target = np.arctan2(self.look_at_y - self.y_target, self.look_at_x - self.x_target)
                
                self.stop_x = None 
                self.stop_y = None
                control_command = [self.x_target, self.y_target, self.z_target, self.yaw_target]
                
            elif gate_in_sight:
                control_command = [self.stop_x, self.stop_y, self.dynamic_z, self.yaw_target]
                
            else:
                self.stop_x = None
                self.stop_y = None
                
                if not self.is_repositioning:
                    self.yaw_target += self.yaw_rate * dt
                    self.yaw_accumulated += self.yaw_rate * dt
                    
                    if self.yaw_accumulated >= 2 * np.pi:
                        self.is_vertical_search_active = True
                    
                    if self.yaw_accumulated >= 4 * np.pi:
                        self.terminal_print("Angle mort détecté. Déplacement de secours de 1.5m.")
                        self.is_repositioning = True
                        current_yaw = sensor_data['yaw']
                        self.reposition_target_x = sensor_data['x_global'] + 1.5 * np.cos(current_yaw)
                        self.reposition_target_y = sensor_data['y_global'] + 1.5 * np.sin(current_yaw)
                        self.yaw_accumulated = 0.0
                        self.is_vertical_search_active = False

                    z_cmd = 1.15
                    if self.is_vertical_search_active:
                        if abs(sensor_data['z_global'] - self.vertical_search_target) < 0.1:
                            self.vertical_search_target = 0.7 if self.vertical_search_target == 2.0 else 2.0
                        z_cmd = self.vertical_search_target
                        
                    control_command = [sensor_data['x_global'], sensor_data['y_global'], z_cmd, self.yaw_target]
                
                else:
                    control_command = [self.reposition_target_x, self.reposition_target_y, sensor_data['z_global'], self.yaw_target]
                    dist_to_repo = np.linalg.norm([
                        sensor_data['x_global'] - self.reposition_target_x, 
                        sensor_data['y_global'] - self.reposition_target_y
                    ])
                    if dist_to_repo < 0.2:
                        self.terminal_print("Nouveau point d'observation atteint. Reprise du scan.")
                        self.is_repositioning = False

        if self.state == LATERAL_TRAVEL:
            if np.linalg.norm(np.array([sensor_data['x_global'] - self.x_target, sensor_data['y_global'] - self.y_target, sensor_data['z_global'] - self.z_target])) < 0.05:
                self.state = DETECT_2
                self.yaw_target = sensor_data['yaw']
                self.terminal_print("Reached lateral travel position, transitioning to DETECT_2 state")
            control_command = [self.x_target, self.y_target, self.z_target, self.yaw_target]

        if self.state == DETECT_2:
            gate_valid = False
            gate_in_sight = False
            img_bgr = cv2.cvtColor(camera_data, cv2.COLOR_BGRA2BGR)
            img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            lower_pink = np.array([140,50,50])
            upper_pink = np.array([160,255,255])
            mask = cv2.inRange(img_hsv, lower_pink, upper_pink)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

            if len(contours) > 0:
                sorted_contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[3], reverse=True)
                
                H = mask.shape[0]
                W = mask.shape[1]
                focal_length = 161.013922282
                real_height = 0.4
                pitch_angle = abs(sensor_data['pitch'])
                valid_contour = None
                
                for c in sorted_contours:
                    x, y, w, h = cv2.boundingRect(c)
                    contour_area = cv2.contourArea(c)
                    
                    if contour_area <= MIN_AREA_PIXELS or h <= MIN_HEIGHT_PIXELS:
                        continue
                        
                    padding = 5
                    touches_top = (y < padding)
                    touches_bottom = (y + h > H - padding)
                    touches_left = (x < padding)
                    touches_right = (x + w > W - padding)
                    is_partially_occluded = touches_top or touches_bottom or touches_left or touches_right

                    screen_area = W * H
                    coverage_ratio = contour_area / screen_area

                    is_dangerously_close = False
                    if is_partially_occluded and coverage_ratio > 0.60: 
                        is_dangerously_close = True
                        is_partially_occluded = False

                    h_corrige = h / max(np.cos(pitch_angle), 0.1)
                    estimated_distance = (focal_length * real_height) / h_corrige
                    
                    if self.memorized_gate_width is not None:
                        largeur_pixels_attendue = (self.memorized_gate_width * focal_length) / estimated_distance
                        u_c = (x + w) - (largeur_pixels_attendue / 2.0)
                    else:
                        u_c = x + w / 2.0

                    drone_x = sensor_data['x_global']
                    drone_y = sensor_data['y_global']
                    drone_yaw = sensor_data['yaw']
                    
                    angle_offset = np.arctan2((u_c - W / 2.0), focal_length)
                    est_gate_yaw = drone_yaw - angle_offset
                    
                    est_gate_x = drone_x + estimated_distance * np.cos(est_gate_yaw)
                    est_gate_y = drone_y + estimated_distance * np.sin(est_gate_yaw)
                    
                    center_x, center_y = 4.0, 4.0
                    angle_rad = np.arctan2(est_gate_y - center_y, est_gate_x - center_x)
                    angle_deg = np.degrees(angle_rad)
                    
                    clock_angle = (360 - angle_deg + 15) % 360
                    expected_center_angle = self.expected_cadrans[self.curr_gate_index] * 30 + 15
                    
                    angle_diff = abs(clock_angle - expected_center_angle)
                    angle_diff = min(angle_diff, 360 - angle_diff)
                    
                    if angle_diff <= 45.0:
                        valid_contour = c
                        best_x, best_y, best_w, best_h = x, y, w, h
                        best_dist = estimated_distance
                        best_est_x, best_est_y = est_gate_x, est_gate_y
                        best_danger = is_dangerously_close
                        best_u_c = u_c
                        break
                        
                if valid_contour is not None:
                    tallest_contour = valid_contour
                    x, y, w, h = best_x, best_y, best_w, best_h
                    estimated_distance = best_dist
                    est_gate_x, est_gate_y = best_est_x, best_est_y
                    is_dangerously_close = best_danger
                    u_c = best_u_c
                    
                    if is_dangerously_close:
                        self.terminal_print("DANGER PROXIMITÉ 2 : Traversée forcée !")
                        self.state = TRAVEL_GATE
                        yaw = sensor_data['yaw']
                        self.exit_x = sensor_data['x_global'] + 1.5 * np.cos(yaw)
                        self.exit_y = sensor_data['y_global'] + 1.5 * np.sin(yaw)
                        self.gate_z_target = sensor_data['z_global']
                        self.gate_yaw_target = yaw
                        self.spline_t = 2.0 
                    else:
                        self.yaw_accumulated = 0.0
                        self.is_vertical_search_active = False

                        self.est_gate_pos_2d_bis[self.curr_gate_index, 0] = est_gate_x
                        self.est_gate_pos_2d_bis[self.curr_gate_index, 1] = est_gate_y
                        
                        gate_in_sight = True 
                        
                        v_c = y + h / 2.0  
                        center_y_img = H / 2.0 
                        if v_c < center_y_img - 20: 
                            self.dynamic_z = sensor_data['z_global'] + 0.1 
                        elif v_c > center_y_img + 20:
                            self.dynamic_z = sensor_data['z_global'] - 0.1
                        else:
                            self.dynamic_z = sensor_data['z_global']
                            
                        if self.stop_x is None: 
                            self.stop_x = sensor_data['x_global']
                            self.stop_y = sensor_data['y_global']
                            
                        global_speed = np.linalg.norm([sensor_data['v_x'], sensor_data['v_y'], sensor_data['v_z']])
                        if global_speed < 0.1: 
                            self.frames_detected += 1
                            if self.frames_detected >= self.req_frames:
                                corners_2d_sorted = self.get_corner_positions_2d_sorted(tallest_contour)

                                quaternion = [sensor_data['q_x'], sensor_data['q_y'], sensor_data['q_z'], sensor_data['q_w']]
                                R_body_to_world = R.from_quat(quaternion).as_matrix()
                                R_cam_to_body = np.array([[0,0,1],[-1,0,0],[0,-1,0]]) 

                                for i, corner in enumerate(corners_2d_sorted):
                                    v_camera_2 = np.array([corner[0] - W/2, corner[1] - H/2, focal_length])
                                    self.s_corners[i] = R_body_to_world @ (R_cam_to_body @ v_camera_2)

                                pos_drone = np.array([sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global']])
                                self.Q = pos_drone + (R_body_to_world @ np.array([0.03, 0, 0.01]))
                                gate_valid = True
                                self.frames_detected = 0
                        else:
                            self.frames_detected = 0 
                else:
                    self.frames_detected = 0 
            else:
                self.frames_detected = 0 

            # --- TRANSITIONS DETECT 2 ---
            if self.state == TRAVEL_GATE:
                control_command = [self.exit_x, self.exit_y, self.gate_z_target, self.gate_yaw_target]
            elif gate_valid:
                self.state = COMPUTE_GATE_POS
                self.terminal_print("Gate 2 detected and validated, transitioning to COMPUTE_GATE_POS")
                self.stop_x = None 
                self.stop_y = None
                control_command = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], self.yaw_target]
            elif gate_in_sight:
                control_command = [self.stop_x, self.stop_y, self.dynamic_z, self.yaw_target]
            else:
                self.stop_x = None
                self.stop_y = None
                
                if not self.is_repositioning:
                    self.yaw_target += self.yaw_rate * dt
                    self.yaw_accumulated += self.yaw_rate * dt
                    
                    if self.yaw_accumulated >= 2 * np.pi:
                        self.is_vertical_search_active = True
                    
                    if self.yaw_accumulated >= 4 * np.pi:
                        self.terminal_print("Angle mort détecté. Déplacement de secours de 1.5m.")
                        self.is_repositioning = True
                        current_yaw = sensor_data['yaw']
                        self.reposition_target_x = sensor_data['x_global'] + 1.5 * np.cos(current_yaw)
                        self.reposition_target_y = sensor_data['y_global'] + 1.5 * np.sin(current_yaw)
                        self.yaw_accumulated = 0.0
                        self.is_vertical_search_active = False

                    z_cmd = 1.15
                    if self.is_vertical_search_active:
                        if abs(sensor_data['z_global'] - self.vertical_search_target) < 0.1:
                            self.vertical_search_target = 0.7 if self.vertical_search_target == 2.0 else 2.0
                        z_cmd = self.vertical_search_target
                        
                    control_command = [sensor_data['x_global'], sensor_data['y_global'], z_cmd, self.yaw_target]
                
                else:
                    control_command = [self.reposition_target_x, self.reposition_target_y, sensor_data['z_global'], self.yaw_target]
                    dist_to_repo = np.linalg.norm([
                        sensor_data['x_global'] - self.reposition_target_x, 
                        sensor_data['y_global'] - self.reposition_target_y
                    ])
                    if dist_to_repo < 0.2:
                        self.terminal_print("Nouveau point d'observation atteint. Reprise du scan.")
                        self.is_repositioning = False

        if self.state == COMPUTE_GATE_POS:
            A = np.zeros((3,2))
            b = np.zeros((3,1))
            corners_3d = np.zeros((4, 3))
            for i in range(len(self.r_corners)):
                A[:,0] = self.r_corners[i]
                A[:,1] = -self.s_corners[i]
                b = self.Q - self.P
                sol = np.linalg.pinv(A) @ b
                lmbda, mu = sol[0], sol[1]
                F = self.P + lmbda * self.r_corners[i]
                G = self.Q + mu * self.s_corners[i]
                corners_3d[i] = (F + G) / 2
            
            H = np.mean(corners_3d, axis=0)

            v_width = corners_3d[1] - corners_3d[0]
            dx_normal = -v_width[1]
            dy_normal = v_width[0]
            gate_yaw = np.arctan2(dy_normal, dx_normal)
            
            # La suite de ton code reste inchangée...
            gate_index = self.curr_gate_index
            self.gate_pos[gate_index,:] = [H[0], H[1], H[2], gate_yaw] # Assuming yaw of gate is 0, you can change this if you have a different method to estimate the gate's yaw

            self.gate_corner[gate_index, :] = corners_3d

            # ... (Début de COMPUTE_GATE_POS inchangé)
            self.curr_gate_index += 1

            # --- MÉMORISATION DES CIBLES ---
            self.gate_yaw_target = gate_yaw
            self.gate_z_target = H[2]
        
            side_margin = 0.05  # 5 cm vers la droite
            offset_x = side_margin * np.sin(gate_yaw)
            offset_y = -side_margin * np.cos(gate_yaw)
            
            # Point d'approche à 0.6m
            app_dist = 0.6
            self.app_x = H[0] - app_dist * np.cos(gate_yaw)
            self.app_y = H[1] - app_dist * np.sin(gate_yaw)
            
            # Centre de la porte
            self.mid_x = H[0] + offset_x
            self.mid_y = H[1] + offset_y
            
            # Point de sortie à 0.6m
            exit_dist = 0.6
            self.exit_x = H[0] + exit_dist * np.cos(gate_yaw)
            self.exit_y = H[1] + exit_dist * np.sin(gate_yaw)

            # --- ON SUPPRIME LES SPLINES ICI ---
            self.travel_phase = 0
            self.state = TRAVEL_GATE
            self.terminal_print("Passage géométrique strict calculé ! Début du vol...")
            
            control_command = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], self.yaw_target]

        if self.state == TRAVEL_GATE:
            
            # --- PHASE 0 : VOLER VERS L'APPROCHE (0.6m devant) ---
            if self.travel_phase == 0:
                self.x_target = self.app_x
                self.y_target = self.app_y
                self.z_target = self.gate_z_target
                self.yaw_target = self.gate_yaw_target 
                
                dist_to_app = np.linalg.norm([
                    sensor_data['x_global'] - self.app_x,
                    sensor_data['y_global'] - self.app_y,
                    sensor_data['z_global'] - self.gate_z_target
                ])
                
                if dist_to_app < 0.10: # Dès qu'on est très proche, on vise le milieu
                    self.travel_phase = 1
                    
            # --- PHASE 1 : FRANCHISSEMENT (Vers le milieu exact) ---
            elif self.travel_phase == 1:
                self.x_target = self.mid_x
                self.y_target = self.mid_y
                self.z_target = self.gate_z_target
                self.yaw_target = self.gate_yaw_target
                
                dist_to_mid = np.linalg.norm([
                    sensor_data['x_global'] - self.mid_x,
                    sensor_data['y_global'] - self.mid_y,
                    sensor_data['z_global'] - self.gate_z_target
                ])
                
                if dist_to_mid < 0.10: # On a passé le cadre, on vise la sortie
                    self.travel_phase = 2
                    
            # --- PHASE 2 : DÉGAGEMENT (0.6m derrière) ---
            elif self.travel_phase == 2:
                self.x_target = self.exit_x
                self.y_target = self.exit_y
                self.z_target = self.gate_z_target
                self.yaw_target = self.gate_yaw_target
                
                dist_to_exit = np.linalg.norm([
                    sensor_data['x_global'] - self.exit_x,
                    sensor_data['y_global'] - self.exit_y,
                    sensor_data['z_global'] - self.gate_z_target
                ])
                
                # On s'assure d'avoir bien dégagé la porte !
                if dist_to_exit < 0.10:
                    self.terminal_print("Porte franchie en phase de détection !")
                    
                    self.stop_x = sensor_data['x_global']
                    self.stop_y = sensor_data['y_global']
                    
                    if self.curr_gate_index < 5:
                        self.state = DETECT_1
                        self.yaw_target = sensor_data['yaw']
                    else:
                        self.state = COMPUTE_PATH
                        
            control_command = [self.x_target, self.y_target, self.z_target, self.yaw_target]
        
        if self.state == COMPUTE_PATH:
            if self.curr_gate_index < 5:
                self.terminal_print("CRITICAL ERROR: Not all gate have been detected")
            self.terminal_print("--- Vérification et Tri des portes ---")
            
            # 1. On utilise le centre connu du circuit
            cx, cy = 4.0, 4.0
            
            # 2. Calcul de l'angle géométrique (atan2) de chaque porte
            angles = np.arctan2(self.gate_pos[:, 1] - cy, self.gate_pos[:, 0] - cx)
            
            # 3. Le tri magique : argsort classe du plus petit au plus grand angle.
            # Grâce à l'axe de Webots, cela correspond EXACTEMENT à l'ordre anti-horaire 
            # en partant de la zone de Take-Off !
            sorted_indices = np.argsort(angles)
            
            # 4. On réécrit nos mémoires dans le bon ordre
            self.gate_pos = self.gate_pos[sorted_indices]
            self.gate_corner = self.gate_corner[sorted_indices]
            
            self.terminal_print(f"Ordre des portes corrigé : {sorted_indices}")
            # ----------------------------------------------
                
            nbr_of_lap = 2
            for lap in range(nbr_of_lap):
                for gate_index in range(5):
                    gate_x = self.gate_pos[gate_index, 0]
                    gate_y = self.gate_pos[gate_index, 1]
                    gate_z = self.gate_pos[gate_index, 2]
                    gate_yaw = self.gate_pos[gate_index, 3]

                    wp_app_x = gate_x - self.approach_distance * np.cos(gate_yaw)
                    wp_app_y = gate_y - self.approach_distance * np.sin(gate_yaw)

                    #before the gate
                    self.racing_waypoints.append([wp_app_x, wp_app_y, gate_z, gate_yaw])
                    #at the gate
                    self.racing_waypoints.append([gate_x, gate_y, gate_z, gate_yaw])
                    #after the gate
                    wp_exit_x = gate_x + self.approach_distance * np.cos(gate_yaw)
                    wp_exit_y = gate_y + self.approach_distance * np.sin(gate_yaw)
                    self.racing_waypoints.append([wp_exit_x, wp_exit_y, gate_z, gate_yaw])

            self.x_target = self.racing_waypoints[0][0]
            self.y_target = self.racing_waypoints[0][1]
            self.z_target = self.racing_waypoints[0][2]
            self.yaw_target = self.racing_waypoints[0][3]
            if self.show_plot:
                # Plot detected gates (position + orientation) and start point
                is_main_thread = threading.current_thread() is threading.main_thread()
                if is_main_thread:
                    fig, ax = plt.subplots(figsize=(8, 8))
                else:
                    fig = Figure(figsize=(8, 8))
                    FigureCanvas(fig)
                    ax = fig.add_subplot(111)

                # Use only initialized gates if needed
                gate_indices = [i for i in range(5) if np.linalg.norm(self.gate_pos[i, :3]) > 1e-9]
                if len(gate_indices) == 0:
                    gate_indices = list(range(5))

                gx = self.gate_pos[gate_indices, 0]
                gy = self.gate_pos[gate_indices, 1]
                gyaw = self.gate_pos[gate_indices, 3]

                # Rotate map content 90 degrees counterclockwise within the 8x8 arena.
                gx_rot = 8.0 - gy
                gy_rot = gx

                # Gate centers
                ax.scatter(gx_rot, gy_rot, c="magenta", s=70, label="Gates")

                est_x = self.est_gate_pos_2d[gate_indices, 0]
                est_y = self.est_gate_pos_2d[gate_indices, 1]
                
                # On applique la même rotation que pour le reste de la carte
                est_x_rot = 8.0 - est_y
                est_y_rot = est_x
                
                # On dessine les estimations sous forme de croix oranges
                ax.scatter(est_x_rot, est_y_rot, c="orange", marker="x", s=60, label="Estimated Gates (2D)")

                est_x_bis = self.est_gate_pos_2d_bis[gate_indices, 0]
                est_y_bis = self.est_gate_pos_2d_bis[gate_indices, 1]

                # On applique la même rotation que pour le reste de la carte
                est_x_rot_bis = 8.0 - est_y_bis
                est_y_rot_bis = est_x_bis
                
                # On dessine les estimations sous forme de croix oranges
                ax.scatter(est_x_rot_bis, est_y_rot_bis, c="red", marker="x", s=60, label="Estimated Gates bis (2D)")

                # Orientation arrows
                u = np.cos(gyaw)
                v = np.sin(gyaw)
                u_rot = -v
                v_rot = u
                ax.quiver(gx_rot, gy_rot, u_rot, v_rot, angles="xy", scale_units="xy", scale=4, color="purple", width=0.004)

                # Gate labels
                for k, i in enumerate(gate_indices):
                    ax.text(gx_rot[k] + 0.05, gy_rot[k] + 0.05, f"G{i+1}", color="black", fontsize=9)

                # Start point
                start_x_rot = 8.0 - self.start_y
                start_y_rot = self.start_x
                ax.scatter(start_x_rot, start_y_rot, c="green", marker="*", s=160, label="Start")
                ax.text(start_x_rot + 0.05, start_y_rot + 0.05, "Start", color="green", fontsize=9)

                ax.set_title("Gate positions and orientations")
                ax.set_xlabel("X [m]")
                ax.set_ylabel("Y [m]")
                ax.set_xlim(0, 8)
                ax.set_ylim(0, 8)
                ax.axis("equal")
                ax.grid(True)
                ax.legend()
                fig.tight_layout()
                if is_main_thread:
                    plt.show(block=False)
                    # plt.pause(0.001)
                else:
                    plot_path = os.path.join(os.path.dirname(__file__), "race_gates_plot.png")
                    fig.savefig(plot_path, dpi=150)
                    self.terminal_print(f"Plot saved to {plot_path} (GUI disabled in planner thread)")
                if is_main_thread:
                    plt.close(fig)
            else:
                self.terminal_print("Plotting disabled (show_plot=False)")
            self.state = RACE
            self.terminal_print("Computed racing path, transitioning to RACE state")
            control_command = [self.x_target, self.y_target, self.z_target, self.yaw_target]

        if self.state == RACE:
            #VERSION SUPER BASIQUE QUI FONCTIONNE MAIS QUI N'EST PAS OPTIMISEE DU TOUT, IL FAUT JUSTE SUIVRE LES WAYPOINTS DANS L'ORDRE, IL N'Y A PAS DE CONTROLEUR AVANCE NI DE PREVISION DES PROCHAINES POSITIONS, C'EST JUSTE UN SUIVI DE CHEMIN BASIQUE
            if self.racing_waypoint_index >= len(self.racing_waypoints):
                control_command = [self.start_x, self.start_y, 1.0, self.start_yaw] # hover at the starting position after finishing
            else:
                #actual waypoint
                self.x_target = self.racing_waypoints[self.racing_waypoint_index][0]
                self.y_target = self.racing_waypoints[self.racing_waypoint_index][1]
                self.z_target = self.racing_waypoints[self.racing_waypoint_index][2]
                self.yaw_target = self.racing_waypoints[self.racing_waypoint_index][3]

                distance = np.linalg.norm(np.array([
                sensor_data['x_global'] - self.x_target, 
                sensor_data['y_global'] - self.y_target, 
                sensor_data['z_global'] - self.z_target
                ]))

                if distance < 0.15:
                    self.racing_waypoint_index += 1
                    if self.racing_waypoint_index < len(self.racing_waypoints):
                        self.terminal_print(f"Reached waypoint {self.racing_waypoint_index}, moving to next waypoint")
                    else:
                        self.terminal_print("Reached final waypoint, race completed")
                        

                control_command = [self.x_target, self.y_target, self.z_target, self.yaw_target]
                
        # --- LISSAGE DE LA POSITION (Carrot-on-a-stick) ---
        # On ne lisse pas si on est en HOVER (on veut monter direct) ou en COMPUTE (calculs en cours)
        if self.state not in [HOVER, COMPUTE_GATE_POS, COMPUTE_PATH]:
            
            # 1. On définit la vitesse globale en fonction de l'état
            if self.state == RACE:
                max_carrot_dist = 0.8  # RAPIDE : Le drone peut "tirer" jusqu'à 80cm d'un coup
            else:
                max_carrot_dist = 0.3  # LENT : Le drone avance doucement (15cm) pendant la détection
            
            cmd_x, cmd_y, cmd_z, cmd_yaw = control_command
            
            curr_x = sensor_data['x_global']
            curr_y = sensor_data['y_global']
            curr_z = sensor_data['z_global']
            
            dx = cmd_x - curr_x
            dy = cmd_y - curr_y
            dz = cmd_z - curr_z
            
            dist = np.linalg.norm([dx, dy, dz])
            
            # 2. Application du lissage
            if dist > max_carrot_dist:
                cmd_x = curr_x + (dx / dist) * max_carrot_dist
                cmd_y = curr_y + (dy / dist) * max_carrot_dist
                cmd_z = curr_z + (dz / dist) * max_carrot_dist
                
            control_command = [cmd_x, cmd_y, cmd_z, cmd_yaw]

        return control_command # Ordered as array with: [pos_x_cmd, pos_y_cmd, pos_z_cmd, yaw_cmd] in meters and radians


# Module-level singleton so main.py can call assignment.get_command() unchanged
_controller = MyAssignment()

def get_command(sensor_data, camera_data, dt):
    return _controller.compute_command(sensor_data, camera_data, dt)
