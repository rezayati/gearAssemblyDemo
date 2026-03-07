import os
import time
import numpy as np
import cv2
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation
from franky import *

# --- 1. CONFIGURATION ---
ROBOT_IP = "192.168.15.33"
# If your board has 9x6 squares, change this to:[8, 5]
CHECKERBOARD = (8, 5)
SQUARE_SIZE = 0.030 # 30mm

# Prepare 3D points of the checkerboard (Z=0)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

def extract_pose_data(robot_state):
    """
    Extracts 3x3 Rotation matrix and 3x1 Translation vector from the robot state.
    Dynamically handles the franky C++ Affine wrapper.
    """
    # Grab the Affine object from the state
    affine = robot_state.O_T_EE
    
    t = None
    R = None

    # 1. Extract Translation (X, Y, Z)
    if hasattr(affine, 'vector'):
        t = affine.vector() if callable(affine.vector) else affine.vector
    elif hasattr(affine, 'translation'):
        t = affine.translation() if callable(affine.translation) else affine.translation
    
    # 2. Extract Rotation (Matrix, Euler, or Quaternion)
    if hasattr(affine, 'rotation'):
        R = affine.rotation() if callable(affine.rotation) else affine.rotation
    elif hasattr(affine, 'angles'): 
        # franky often exposes .angles() for ZYX Euler angles
        euler = affine.angles() if callable(affine.angles) else affine.angles
        R = Rotation.from_euler('xyz', euler).as_matrix()
    elif hasattr(affine, 'quaternion'):
        q = affine.quaternion() if callable(affine.quaternion) else affine.quaternion
        R = Rotation.from_quat(q).as_matrix()

    # If we successfully found both, format them for OpenCV and return
    if t is not None and R is not None:
        return np.array(R).reshape(3, 3), np.array(t).reshape(3, 1)
        
    # Fallback: If the API changed completely, print out what IS available so we can see it
    raise ValueError(f"Could not extract pose. Available attributes on Affine: {dir(affine)}")


def generate_calibration_offsets():
    """Generates a list of (dx, dy, dz, roll_offset, pitch_offset, yaw_offset)"""
    offsets = []
    angle = 0.25 
    dist = 0.04 
    
    # 1. Center, but tilted
    offsets.append((0, 0, 0, angle, 0, 0))
    offsets.append((0, 0, 0, -angle, 0, 0))
    offsets.append((0, 0, 0, 0, angle, 0))
    offsets.append((0, 0, 0, 0, -angle, 0))
    offsets.append((0, 0, 0, 0, 0, angle))
    
    # 2. Translated and tilted
    offsets.append((dist, dist, 0, angle, 0, 0))
    offsets.append((-dist, -dist, 0, -angle, 0, 0))
    offsets.append((dist, -dist, 0, 0, angle, 0))
    offsets.append((-dist, dist, 0, 0, -angle, 0))
    
    # 3. Z height changes
    offsets.append((0, 0, dist, angle/2, angle/2, 0))
    offsets.append((0, 0, -dist, -angle/2, -angle/2, 0))
    
    return offsets

def main():
    print("Connecting to robot...")
    robot = Robot(ROBOT_IP)
    robot.recover_from_errors()
    robot.relative_dynamics_factor = 0.03 

    print("Starting RealSense camera...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    camera_matrix = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
    dist_coeffs = np.zeros(5)

    R_gripper2base_list = []
    t_gripper2base_list = []
    R_target2cam_list = []
    t_target2cam_list = []

    # Save the initial RobotPose object so we can command the robot to return home later
    start_pose_obj = robot.current_pose
    
    # Extract the mathematical pose directly from the raw libfranka state
    start_R, start_t = extract_pose_data(robot.state)
    start_euler = Rotation.from_matrix(start_R).as_euler('xyz')

    offsets = generate_calibration_offsets()
    print(f"Starting automated calibration sequence ({len(offsets)} poses)...")

    try:
        for i, (dx, dy, dz, droll, dpitch, dyaw) in enumerate(offsets):
            print(f"Moving to pose {i+1}/{len(offsets)}...")
            
            # Calculate new absolute pose
            new_t = start_t.flatten() + np.array([dx, dy, dz])
            new_euler = start_euler + np.array([droll, dpitch, dyaw])
            new_quat = Rotation.from_euler('xyz', new_euler).as_quat()
            
            target_affine = Affine(new_t.tolist(), new_quat.tolist())
            robot.move(CartesianMotion(target_affine))
            
            time.sleep(1.0) # Wait for arm to settle
            
            for _ in range(10): pipeline.wait_for_frames() # Flush old frames
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame: continue

            img = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            cv2.imshow("Calibration View", img)
            cv2.waitKey(1) # Required to refresh the window

            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

            if ret:
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                
                _, rvec, tvec = cv2.solvePnP(objp, corners2, camera_matrix, dist_coeffs)
                R_t2c, _ = cv2.Rodrigues(rvec)
                
                # Get EXACT Robot Pose at this moment from raw state
                current_R, current_t = extract_pose_data(robot.state)
                
                R_target2cam_list.append(R_t2c)
                t_target2cam_list.append(tvec)
                R_gripper2base_list.append(current_R)
                t_gripper2base_list.append(current_t)
                print("  -> Capture successful!")
            else:
                print("  -> Checkerboard not fully visible, skipping.")

    finally:
        print("Returning to Home position...")
        try:
            # Clear the reflex error so the robot is allowed to move again
            robot.recover_from_errors() 
            robot.move(CartesianMotion(start_pose_obj))
        except Exception as e:
            print(f"Could not return home automatically: {e}")
            
        pipeline.stop()
        cv2.destroyAllWindows() # Close the video window

    if len(R_gripper2base_list) < 5:
        print("Error: Not enough valid captures. Adjust the lighting or starting height.")
        return

    print("Computing Tsai-Lenz Hand-Eye Calibration...")
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base_list, t_gripper2base_list,
        R_target2cam_list, t_target2cam_list,
        method=cv2.CALIB_HAND_EYE_TSAI
    )

    T_cam2gripper = np.eye(4)
    T_cam2gripper[:3, :3] = R_cam2gripper
    T_cam2gripper[:3, 3] = t_cam2gripper.flatten()
    
    # Calculate the absolute path to your config folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(script_dir, "..", "config")
    
    # Ensure the config folder exists
    os.makedirs(config_dir, exist_ok=True)
    
    save_path = os.path.join(config_dir, "calibration.npy")
    
    np.save(save_path, T_cam2gripper)
    print(f"\n✅ Calibration successful and saved to: {save_path}")
    print("Transformation Matrix:\n", T_cam2gripper)

if __name__ == "__main__":
    main()