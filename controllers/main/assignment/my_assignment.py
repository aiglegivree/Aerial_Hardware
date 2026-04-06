import numpy as np
import time
import cv2

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
DETECT = 2
TRAVEL = 3
COMPUTE_PATH = 4
RACE = 5

class MyAssignment:
    def __init__(self):
        # ---- INITIALISE YOUR VARIABLES HERE ----
        self.state = INIT
        self.start_x = 0
        self.start_y = 0
        self.start_yaw = 0
        self.gate_pos = np.zeros((5, 4)) # five gates with x,y,z pos and yaw angle of the center of the gate
        self.yaw_rate = np.pi/4 # yaw rate in radians per second

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

        if self.state == HOVER:
            control_command = [self.start_x, self.start_y, 1.0, self.start_yaw]
            if abs(sensor_data['z_global'] - 1.0) < 0.05:
                self.state = DETECT
        
        if self.state == DETECT:
            # Here you would add your object detection code using camera_data
            # For example, you could use OpenCV to process the camera image and detect the target object
            # If the object is detected, you would then transition to the TRAVEL state
            gate_detected = False # This should be set to True if the gate is detected
            ##detection logic to implement here, which sets gate_detected to True if the gate is detected and also updates self.gate_pos with the position of the detected gate

            if not gate_detected:
                new_yaw = sensor_data['yaw'] + self.yaw_rate * dt
                control_command = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], new_yaw]   
            else:
                self.state = TRAVEL

        if self.state == TRAVEL:
            #not yet implemented
            control_command = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw']] # You can set the control command to the current position and yaw while you are implementing the detection logic
        return control_command # Ordered as array with: [pos_x_cmd, pos_y_cmd, pos_z_cmd, yaw_cmd] in meters and radians


# Module-level singleton so main.py can call assignment.get_command() unchanged
_controller = MyAssignment()

def get_command(sensor_data, camera_data, dt):
    return _controller.compute_command(sensor_data, camera_data, dt)
