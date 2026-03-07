import time
import numpy as np
import cv2
import pyrealsense2 as rs
import os
from scipy.spatial.transform import Rotation
from franky import *

ROBOT_IP = "192.168.15.33"

def get_T_flange_to_base(robot_state):
    """
    Extracts the 4x4 transformation matrix from the robot state.
    Dynamically handles the franky C++ Affine wrapper.
    """
    affine = robot_state.O_T_EE
    
    t = None
    R = None

    # 1. Extract Translation
    if hasattr(affine, 'vector'):
        t = affine.vector() if callable(affine.vector) else affine.vector
    elif hasattr(affine, 'translation'):
        t = affine.translation() if callable(affine.translation) else affine.translation
    
    # 2. Extract Rotation
    if hasattr(affine, 'rotation'):
        R = affine.rotation() if callable(affine.rotation) else affine.rotation
    elif hasattr(affine, 'angles'): 
        euler = affine.angles() if callable(affine.angles) else affine.angles
        R = Rotation.from_euler('xyz', euler).as_matrix()
    elif hasattr(affine, 'quaternion'):
        q = affine.quaternion() if callable(affine.quaternion) else affine.quaternion
        R = Rotation.from_quat(q).as_matrix()

    if t is not None and R is not None:
        # Build the 4x4 Transformation Matrix
        T = np.eye(4)
        T[:3, :3] = np.array(R).reshape(3, 3)
        T[:3, 3] = np.array(t).flatten()
        return T
        
    raise ValueError(f"Could not extract pose. Available attributes on Affine: {dir(affine)}")

def main():
    print("Loading calibration matrix...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    calib_path = os.path.join(script_dir, "..", "config", "calibration.npy")
    
    if not os.path.exists(calib_path):
        print(f"Error: Could not find {calib_path}")
        return
    T_cam2gripper = np.load(calib_path)

    print("Connecting to robot...")
    robot = Robot(ROBOT_IP)
    gripper = Gripper(ROBOT_IP)
    robot.recover_from_errors()
    robot.relative_dynamics_factor = 0.03 

    # 1. RECORD GROUND TRUTH
    # We assume you manually jogged the robot so the gripper is perfectly centered on the gear.
    T_start = get_T_flange_to_base(robot.state)
    ground_truth_x = T_start[0, 3]
    ground_truth_y = T_start[1, 3]
    # We don't record Z for the gear because the camera sees the TOP of the gear, 
    # but the gripper flange Z is slightly different depending on your finger length.
    
    print("\n--- GROUND TRUTH RECORDED ---")
    print(f"Robot currently at: X={ground_truth_x:.4f}, Y={ground_truth_y:.4f}")
    print("Opening gripper and moving 30cm straight up...")

    # 2. MOVE UP
    gripper.move(0.08, 0.1) # Open gripper so we don't carry the gear with us

    # Calculate the absolute target position (Current Z + 30cm)
    target_x = ground_truth_x
    target_y = ground_truth_y
    target_z = T_start[2, 3] + 0.30
    
    # Extract the current rotation and convert to a quaternion [x, y, z, w]
    rot_matrix = T_start[:3, :3]
    current_quat = Rotation.from_matrix(rot_matrix).as_quat().tolist()
    
    # Move using standard CartesianMotion
    target_affine = Affine([target_x, target_y, target_z], current_quat)
    robot.move(CartesianMotion(target_affine))
    time.sleep(1.0) # Wait for arm to stop wobbling

    # 3. START VISION
    print("\nStarting camera...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(config)
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    try:
        print("Waiting for camera to adjust to lighting...")
        for _ in range(30): pipeline.wait_for_frames()

        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        img = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect the gear (Adjust param1/param2 if it struggles to see the metal gear)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=15, maxRadius=80)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            x, y, r = circles[0, 0]
            
            # Show the detection
            cv2.circle(img, (x, y), r, (0, 255, 0), 2)
            cv2.imshow("Detected Gear", img)
            cv2.waitKey(1000)

            # 4. MATH: PIXEL -> CAMERA FRAME -> BASE FRAME
            depth = depth_frame.get_distance(x, y)
            point_cam = rs.rs2_deproject_pixel_to_point(intr, [x, y], depth)
            P_cam = np.array([point_cam[0], point_cam[1], point_cam[2], 1.0])
            
            T_gripper2base = get_T_flange_to_base(robot.state)
            P_base = T_gripper2base @ T_cam2gripper @ P_cam
            
            calc_x, calc_y, calc_z = P_base[0], P_base[1], P_base[2]
            
            # 5. COMPARE RESULTS
            error_x = abs(ground_truth_x - calc_x)
            error_y = abs(ground_truth_y - calc_y)
            
            print("\n--- TEST RESULTS ---")
            print(f"Ground Truth X: {ground_truth_x:.4f} m | Calculated X: {calc_x:.4f} m | Error: {error_x*1000:.1f} mm")
            print(f"Ground Truth Y: {ground_truth_y:.4f} m | Calculated Y: {calc_y:.4f} m | Error: {error_y*1000:.1f} mm")
            
            if error_x < 0.005 and error_y < 0.005:
                print("✅ Math is perfect! (Error is under 5mm)")
            else:
                print("⚠️ Error is larger than 5mm. We might need to adjust the circle detection center.")

        else:
            print("No gear detected! Try adjusting the lighting.")
            cv2.imshow("Camera View", img)
            cv2.waitKey(3000)

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()