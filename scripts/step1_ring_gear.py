import os
import time
import json
import numpy as np
import cv2
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation
from franky import *
import traceback

# --- NEW GEMINI SETUP ---
from google import genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key: raise ValueError("API Key not found!")

# The new SDK uses a Client object
ai_client = genai.Client(api_key=api_key)

ROBOT_IP = "192.168.15.33"

def get_T_flange_to_base(robot_state):
    """Safely extracts 4x4 matrix from Franky's C++ Affine wrapper."""
    affine = robot_state.O_T_EE
    if hasattr(affine, 'vector'): t = affine.vector() if callable(affine.vector) else affine.vector
    elif hasattr(affine, 'translation'): t = affine.translation() if callable(affine.translation) else affine.translation
    
    if hasattr(affine, 'rotation'): R = affine.rotation() if callable(affine.rotation) else affine.rotation
    elif hasattr(affine, 'angles'): R = Rotation.from_euler('xyz', affine.angles() if callable(affine.angles) else affine.angles).as_matrix()
    elif hasattr(affine, 'quaternion'): R = Rotation.from_quat(affine.quaternion() if callable(affine.quaternion) else affine.quaternion).as_matrix()

    T = np.eye(4)
    T[:3, :3] = np.array(R).reshape(3, 3)
    T[:3, 3] = np.array(t).flatten()
    return T

def ask_gemini_for_ring_gear(image_bgr):
    """Sends the frame to Gemini using the NEW genai SDK."""
    rgb_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    import PIL.Image
    pil_img = PIL.Image.fromarray(rgb_img)
    prompt = """
    Look at this planetary gear assembly table. The image is 640x480.
    Find the exact center of the large dark 'ring gear'. 
    It is the massive, dark gray circular piece.
    It has teeth on the inside.
    Do NOT select the dark square shadows or the metal frame on the far left.
    Return strictly a JSON object with pixel coordinates [x, y].
    Example: {"ring_gear_center": [320, 240]}
    """
    
    # New syntax for generating content
    response = ai_client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[prompt, pil_img]
    )
    return json.loads(response.text.replace('```json', '').replace('```', '').strip())

def refine_center_with_opencv(image_bgr, ai_x, ai_y):
    """Uses classic CV with a localized mask to snap to the exact mathematical center."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    mask = np.zeros_like(gray)
    cv2.circle(mask, (ai_x, ai_y), 150, 255, -1) 
    gray_masked = cv2.bitwise_or(gray, cv2.bitwise_not(mask))
    
    _, thresh = cv2.threshold(gray_masked, 60, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_x, best_y = ai_x, ai_y
    best_contour = None
    max_area = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1500 and area > max_area: 
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                best_x = int(M["m10"] / M["m00"])
                best_y = int(M["m01"] / M["m00"])
                best_contour = cnt
                max_area = area
                    
    return best_x, best_y, best_contour

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    T_cam2gripper = np.load(os.path.join(script_dir, "..", "config", "calibration.npy"))

    robot = Robot(ROBOT_IP)
    robot.recover_from_errors()
    robot.relative_dynamics_factor = 0.05

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config) 
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    try:
        print("\n👀 Live Feed! Press 'u' to jog up, 'y' to accept, 'q' to quit.")
        final_img = None
        while True:
            frames = pipeline.wait_for_frames()
            img = np.asanyarray(frames.get_color_frame().get_data())
            cv2.imshow("Robot View", img)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('u'):
                T_curr = get_T_flange_to_base(robot.state)
                q_curr = Rotation.from_matrix(T_curr[:3, :3]).as_quat().tolist()
                robot.move(CartesianMotion(Affine([T_curr[0,3], T_curr[1,3], T_curr[2,3] + 0.10], q_curr)))
                time.sleep(0.5)
            elif key == ord('y'):
                final_img = img.copy()
                cv2.destroyAllWindows()
                break
            elif key == ord('q'): return

        print("🧠 Asking Gemini for initial guess...")
        targets = ask_gemini_for_ring_gear(final_img)
        ai_x, ai_y = int(targets['ring_gear_center'][0]), int(targets['ring_gear_center'][1])
        print(f"🤖 Gemini's eyeball guess: [{ai_x}, {ai_y}]")

        px_x, px_y, found_contour = refine_center_with_opencv(final_img, ai_x, ai_y)
        print(f"🎯 OpenCV mathematically refined center: [{px_x}, {px_y}]")
        
        debug_img = final_img.copy()
        if found_contour is not None:
            cv2.drawContours(debug_img, [found_contour], -1, (255, 0, 0), 2)
            
        cv2.drawMarker(debug_img, (ai_x, ai_y), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
        cv2.putText(debug_img, "AI Guess", (ai_x - 30, ai_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.circle(debug_img, (px_x, px_y), 5, (0, 255, 0), -1)
        cv2.circle(debug_img, (px_x, px_y), 15, (0, 255, 0), 2)
        cv2.putText(debug_img, "True Center", (px_x + 20, px_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        debug_path = os.path.join(script_dir, "step1_ring_gear.jpg")
        cv2.imwrite(debug_path, debug_img)
        print(f"📸 Visual receipt saved to: {debug_path}")
        
        T_gripper2base = get_T_flange_to_base(robot.state)
        T_cam_in_base = T_gripper2base @ T_cam2gripper
        
        cam_pos = T_cam_in_base[:3, 3]
        cam_rot = T_cam_in_base[:3, :3]
        
        ray_cam = rs.rs2_deproject_pixel_to_point(intr, [px_x, px_y], 1.0)
        ray_base = cam_rot @ np.array(ray_cam) 
        
        if ray_base[2] == 0: raise ValueError("Camera looking parallel to table!")
        t = (0.0 - cam_pos[2]) / ray_base[2]
        target_point = cam_pos + (t * ray_base)
        
        target_gear_x, target_gear_y = target_point[0], target_point[1]

        print("\n--- STEP 1 COMPLETE ---")
        print(f"🎯 Rock-Solid Ring Gear Coordinates: X={target_gear_x:.4f}, Y={target_gear_y:.4f}")

        state_path = os.path.join(script_dir, "..", "config", "assembly_state.json")
        os.makedirs(os.path.dirname(state_path), exist_ok=True)
        with open(state_path, "w") as f:
            json.dump({"ring_gear_x": target_gear_x, "ring_gear_y": target_gear_y}, f)
        print(f"💾 Coordinates saved!")

    except Exception as e:
        traceback.print_exc()
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()