import numpy as np

# Create a rotation matrix from the world frame to the body frame using euler angles
def euler2rotmat(euler_angles):

    R = np.eye(3)

    # Here you need to implement the rotation matrix
    # First calculate the rotation matrix for each angle (roll, pitch, yaw)
    # Then multiply the matrices together to get the total rotation matrix
    # Inputs:
    #           euler_angles: A list of 3 Euler angles [roll, pitch, yaw] in radians
    # Outputs:
    #           R: A 3x3 numpy array that represents the rotation matrix of the euler angles

    # --- YOUR CODE HERE ---
    roll, ptich, yaw = euler_angles
    R_roll = np.array([[1.0, 0.0,0.0],[0.0, np.cos(roll), -np.sin(roll)],[0.0, np.sin(roll), np.cos(roll)]])
    R_pitch = np.array([[np.cos(pitch), 0.0, np.sin(pitch)],[0.0,1.0,0.0],[-np.sin(pitch), 0.0, np.cos(pitch)]])
    R_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0.0],[np.sin(yaw), np.cos(yaw), 0.0],[0.0,0.0,1.0]])

    R = R_yaw @ R_pitch @ R_roll
    return R


# Rotate the control commands from the inertial reference frame to the body reference frame
def rot_inertial2body(control_commands, euler_angles, quaternion):

    # Here you need to rotate the control commands from the inertial reference frame to the body reference frame
    # You should use the euler2rotmat function to get the rotation matrix
    # Keep in mind that you only want to rotate the velocity commands, which are the first two elements of the control_commands array
    # Think carefully about which direction you need to perform the rotation

    # Inputs:
    #           control_commands: A list of 4 control commands [vel_x, vel_y, altitude, yaw_rate] in the inertial reference frame
    #           euler_angles: A list of 3 Euler angles [roll, pitch, yaw] in radians
    #           quaternion: A list of 4 elements [x, y, z, w] representing the quaternion of the drone
    # Outputs:
    #           control_commands: A list of 4 control commands [vel_x, vel_y, altitude, yaw_rate] in the body reference frame

    # --- YOUR CODE HERE ---

    vel_inertial = np.array([control_commands[0], control_commands[1]])
    R = euler2rotmat(euler_angles)
    # vel_body = 

    # control_commands = 

    # --- SAMPLE SOLUTION ---
    vel_inertial = np.array([control_commands[0], control_commands[1], 0.0])
    R = euler2rotmat(euler_angles)
    vel_body = R.T @ vel_inertial

    control_commands = [vel_body[0], vel_body[1], control_commands[2], control_commands[3]]

    return control_commands