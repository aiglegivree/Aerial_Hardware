import numpy as np
import time
import cv2
from scipy.spatial.transform import Rotation as R

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


MIN_HEIGHT_PIXELS = 20

class MyAssignment:
    def __init__(self):
        # ---- INITIALISE YOUR VARIABLES HERE ----
        self.state = INIT
        self.start_x = 0
        self.start_y = 0
        self.start_yaw = 0
        self.gate_pos = np.zeros((5, 4)) # x, y, z, yaw for each gate (5)
        self.gate_corner = np.zeros((5, 4, 3)) # Assuming 4 corners for each gate, you can change this if your gate has a different number of corners
        self.yaw_rate = np.pi/4 # yaw rate in radians per second

        self.P = None
        self.r = None
        self.Q = None
        self.s = None

        self.is_approching_gate = True
        self.approach_distance = 0.3 # distance from which the drone should start approaching the gate, you can adjust this based on your needs

        self.x_target = None
        self.y_target = None
        self.z_target = None
        self.yaw_target = None

        self.curr_gate_index = 0

        self.racing_waypoints = []
        self.racing_waypoint_index = 0

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
            print("Initialized, transitioning to HOVER state")

        if self.state == HOVER:
            control_command = [self.start_x, self.start_y, 1.15, self.start_yaw]
            if abs(sensor_data['z_global'] - 1.15) < 0.05:
                self.state = DETECT_1
                self.yaw_target = sensor_data['yaw']
                print("Reached hover altitude, transitioning to DETECT_1 state")
        
        if self.state == DETECT_1:
            # If the object is detected, you would then transition to the LATERAL_TRAVEL state
            gate_detected = False # This should be set to True if the gate is detected
            img_bgr = cv2.cvtColor(camera_data, cv2.COLOR_BGRA2BGR)
            img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            # Define the lower and upper bounds for the color of the gate in HSV color space
            lower_pink = np.array([140,50,50])
            upper_pink = np.array([160,255,255])
            # Create a mask using the defined color bounds
            mask = cv2.inRange(img_hsv, lower_pink, upper_pink)
            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

            
            if len(contours) > 0:
                largest_contour = max(contours, key=cv2.contourArea)
                H = mask.shape[0]
                W = mask.shape[1]

                padding = 40

                #to see if the detected contour touches the edge of the image, 
                touches_top = np.any(mask[:padding, :] > 0)      
                touches_bottom = np.any(mask[H-padding:, :] > 0)  
                touches_left = np.any(mask[:, :padding] > 0)      
                touches_right = np.any(mask[:, W-padding:] > 0)

                is_partially_occluded = touches_top or touches_bottom or touches_left or touches_right

                x,y,w,h = cv2.boundingRect(largest_contour)

                if cv2.contourArea(largest_contour) > 100 and not is_partially_occluded and h > MIN_HEIGHT_PIXELS:
                    u_pixel = x + w/2
                    v_pixel = y + h/2

                    vect_x = u_pixel - camera_data.shape[1]/2
                    vect_y = v_pixel - camera_data.shape[0]/2
                    vect_z = 161.013922282 #this is the focal length of the camera in pixels from the PDF
                    v_camera = np.array([vect_x, vect_y, vect_z])
                    R_cam_to_body = np.array([[0,0,1],[-1,0,0],[0,-1,0]]) # Rotation matrix from camera frame to body frame
                    v_body = R_cam_to_body @ v_camera

                    quaternion = [sensor_data['q_x'], sensor_data['q_y'], sensor_data['q_z'], sensor_data['q_w']]
                    R_body_to_world = R.from_quat(quaternion).as_matrix()
                    self.r = R_body_to_world @ v_body

                    cam_offset_body = np.array([0.03, 0, 0.01])
                    pos_drone = np.array([sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global']])
                    self.P = pos_drone + (R_body_to_world @ cam_offset_body)
                    gate_detected = True

            if not gate_detected:
                self.yaw_target += self.yaw_rate * dt
                control_command = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], self.yaw_target]   
            else:
                self.state = LATERAL_TRAVEL
                print("Gate detected, transitioning to LATERAL_TRAVEL state")
                yaw = sensor_data['yaw']
                lateral_travel = 0.6
                self.x_target = sensor_data['x_global'] + lateral_travel * np.sin(yaw)
                self.y_target = sensor_data['y_global'] - lateral_travel * np.cos(yaw)
                self.z_target = sensor_data['z_global']
                self.yaw_target = yaw
                control_command = [self.x_target, self.y_target, self.z_target, self.yaw_target]

        if self.state == LATERAL_TRAVEL:
            if np.linalg.norm(np.array([sensor_data['x_global'] - self.x_target, sensor_data['y_global'] - self.y_target, sensor_data['z_global'] - self.z_target])) < 0.05:
                self.state = DETECT_2
                self.yaw_target = sensor_data['yaw']
                print("Reached lateral travel position, transitioning to DETECT_2 state")
            control_command = [self.x_target, self.y_target, self.z_target, self.yaw_target]

        if self.state == DETECT_2:
            gate_detected = False # This should be set to True if the gate is detected
            img_bgr = cv2.cvtColor(camera_data, cv2.COLOR_BGRA2BGR)
            img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            # Define the lower and upper bounds for the color of the gate in HSV color space
            lower_pink = np.array([140,50,50])
            upper_pink = np.array([160,255,255])
            # Create a mask using the defined color bounds
            mask = cv2.inRange(img_hsv, lower_pink, upper_pink)
            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

            if len(contours) > 0:
                largest_contour = max(contours, key=cv2.contourArea)
                H = mask.shape[0]
                W = mask.shape[1]

                #to see if the detected contour touches the edge of the image, 
                touches_top = np.any(mask[0, :] > 0)
                touches_bottom = np.any(mask[H-1, :] > 0)
                touches_left = np.any(mask[:, 0] > 0)
                touches_right = np.any(mask[:, W-1] > 0)

                is_partially_occluded = touches_top or touches_bottom or touches_left or touches_right

                x,y,w,h = cv2.boundingRect(largest_contour)

                if cv2.contourArea(largest_contour) > 100 and not is_partially_occluded and h > MIN_HEIGHT_PIXELS:
                    u_pixel = x + w/2
                    v_pixel = y + h/2

                    vect_x = u_pixel - camera_data.shape[1]/2
                    vect_y = v_pixel - camera_data.shape[0]/2
                    vect_z = 161.013922282 #this is the focal length of the camera in pixels from the PDF
                    v_camera_2 = np.array([vect_x, vect_y, vect_z])
                    R_cam_to_body = np.array([[0,0,1],[-1,0,0],[0,-1,0]]) # Rotation matrix from camera frame to body frame
                    v_body_2 = R_cam_to_body @ v_camera_2

                    quaternion = [sensor_data['q_x'], sensor_data['q_y'], sensor_data['q_z'], sensor_data['q_w']]
                    R_body_to_world = R.from_quat(quaternion).as_matrix()
                    self.s = R_body_to_world @ v_body_2

                    cam_offset_body = np.array([0.03, 0, 0.01])
                    pos_drone = np.array([sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global']])
                    self.Q = pos_drone + (R_body_to_world @ cam_offset_body)
                    gate_detected = True

            if not gate_detected:
                self.yaw_target += 0.5*self.yaw_rate * dt
                control_command = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], self.yaw_target]
            else:
                self.state = COMPUTE_GATE_POS
                print("Gate detected, transitioning to COMPUTE_GATE_POS state")
                control_command = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], self.yaw_target]

        if self.state == COMPUTE_GATE_POS:
            A = np.zeros((3,2))
            b = np.zeros((3,1))
            A[:,0] = self.r
            A[:,1] = -self.s
            b = self.Q - self.P
            sol = np.linalg.pinv(A) @ b
            lmbda, mu = sol[0], sol[1]
            F = self.P + lmbda * self.r
            G = self.Q + mu * self.s
            H = (F + G) / 2
            dx = H[0] - sensor_data['x_global']
            dy = H[1] - sensor_data['y_global']
            gate_yaw = np.arctan2(dy, dx) 
            gate_index = self.curr_gate_index
            self.gate_pos[gate_index,:] = [H[0], H[1], H[2], gate_yaw] # Assuming yaw of gate is 0, you can change this if you have a different method to estimate the gate's yaw
            self.curr_gate_index += 1
            self.x_target = H[0]
            self.y_target = H[1]
            self.z_target = H[2]
            self.yaw_target = gate_yaw
            self.state = TRAVEL_GATE
            print(f"Computed gate position: {H}, transitioning to TRAVEL_GATE state")

        
        if self.state == TRAVEL_GATE:
            if self.is_approching_gate:
                self.x_target = self.gate_pos[self.curr_gate_index-1,0] - self.approach_distance * np.cos(self.gate_pos[self.curr_gate_index-1,3])
                self.y_target = self.gate_pos[self.curr_gate_index-1,1] - self.approach_distance * np.sin(self.gate_pos[self.curr_gate_index-1,3])
                self.z_target = self.gate_pos[self.curr_gate_index-1,2]
                self.yaw_target = self.gate_pos[self.curr_gate_index-1,3]
            else:
                self.x_target = self.gate_pos[self.curr_gate_index-1,0] + self.approach_distance * np.cos(self.gate_pos[self.curr_gate_index-1,3])
                self.y_target = self.gate_pos[self.curr_gate_index-1,1] + self.approach_distance * np.sin(self.gate_pos[self.curr_gate_index-1,3])
                self.z_target = self.gate_pos[self.curr_gate_index-1,2]
                self.yaw_target = self.gate_pos[self.curr_gate_index-1,3]
            if np.linalg.norm(np.array([sensor_data['x_global'] - self.x_target, sensor_data['y_global'] - self.y_target, sensor_data['z_global'] - self.z_target])) < 0.1:
                if self.is_approching_gate:
                    self.is_approching_gate = False
                    print("Reached approach position, now passing through the gate")
                elif self.curr_gate_index < 5:
                    self.state = DETECT_1
                    self.is_approching_gate = True
                    self.yaw_target = sensor_data['yaw']
                    print("Reached gate position, transitioning back to DETECT_1 state for next gate")
                else:
                    self.state = COMPUTE_PATH
                    print("Reached final gate position, transitioning to COMPUTE_PATH state")

            control_command = [self.x_target, self.y_target, self.z_target, self.yaw_target]
        
        if self.state == COMPUTE_PATH:
            #PAS ECNORE FAITE
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
            
            self.state = RACE
            print("Computed racing path, transitioning to RACE state")
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

                if distance < 0.4:
                    self.racing_waypoint_index += 1
                    if self.racing_waypoint_index < len(self.racing_waypoints):
                        print(f"Reached waypoint {self.racing_waypoint_index}, moving to next waypoint")
                    else:
                        print("Reached final waypoint, race completed")
                
                control_command = [self.x_target, self.y_target, self.z_target, self.yaw_target]

        return control_command # Ordered as array with: [pos_x_cmd, pos_y_cmd, pos_z_cmd, yaw_cmd] in meters and radians


# Module-level singleton so main.py can call assignment.get_command() unchanged
_controller = MyAssignment()

def get_command(sensor_data, camera_data, dt):
    return _controller.compute_command(sensor_data, camera_data, dt)
