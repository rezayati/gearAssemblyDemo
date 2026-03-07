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
ai_client = genai.Client(api_key=api_key)

# --- CONFIGURATION ---
ROBOT_IP = "192.168.15.33"
DOWN_QUAT = Rotation.from_euler("xyz", [np.pi, 0, 0]).as_quat().tolist()
PLANETARY_RADIUS = 0.06      
HANDLE_DIAMETER = 0.022      
HANDLE_PICK_Z = 0.055        
GEAR_DROP_Z = 0.025          
HOVER_Z = 0.25               

def get_T_flange_to_base(robot_state):
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

def get_robust_red_mask(image_bgr):
    """
    Surgical Reality Filter: Isolates deep red and physically severs 
    connections to the orange triangle using morphology.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    # Generous red hue (0-8 and 170-180) to ignore shadows, but strict enough to miss orange (10-25)
    mask1 = cv2.inRange(hsv, np.array([0, 100, 50]), np.array([8, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([170, 100, 50]), np.array([180, 255, 255]))
    mask = cv2.bitwise_or(mask1, mask2)
    
    # SURGICAL SEPARATION: Erode and dilate to snap the thin pixel bridge to the orange triangle
    kernel = np.ones((7,7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    return mask

def ask_gemini_for_red_gears(image_bgr, num_gears=3):
    rgb_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    import PIL.Image
    pil_img = PIL.Image.fromarray(rgb_img)
    prompt = f"""
    Look at this planetary gear assembly table. The image is 640x480.
    Find {num_gears} red gears resting in their trays on the table.
    Return strictly a JSON object containing a list of exactly {num_gears} pixel coordinate pairs [x, y], showing the CENTER of each gear.
    Order: Gear 1 (top), Gear 2 (bottom left), Gear 3 (bottom right).
    If you cannot clearly find {num_gears} red gears, return an empty list: {{"red_gears": []}}
    Do not include any conversational text.
    Example format: {{"red_gears": [[150, 200], [300, 250], [400, 100]]}}
    """
    response = ai_client.models.generate_content(model='gemini-2.5-flash', contents=[prompt, pil_img])
    raw_text = response.text.replace('```json', '').replace('```', '').strip()
    try: return json.loads(raw_text)
    except Exception: return {"red_gears": []}

def refine_gear_center(image_bgr, ai_x, ai_y):
    """Snaps AI's hallucinated guess to the physically nearest REAL gear."""
    mask = get_robust_red_mask(image_bgr)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_x, best_y, best_contour, min_dist = ai_x, ai_y, None, float('inf')
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 800: 
            # GEOMETRY CHECK: Bounding box must be roughly square (circles are square)
            w, h = cv2.boundingRect(cnt)[2:]
            if 0.5 < (w / float(h)) < 2.0:
                (x, y), _ = cv2.minEnclosingCircle(cnt)
                # Find the true gear closest to Gemini's guess
                dist = np.sqrt((x - ai_x)**2 + (y - ai_y)**2)
                if dist < min_dist:
                    min_dist, best_x, best_y, best_contour = dist, int(x), int(y), cnt
                    
    return best_x, best_y, best_contour

def find_gear_local(image_bgr):
    """Local precision snap. Ignores everything except the gear under the lens."""
    mask = get_robust_red_mask(image_bgr)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_x, best_y, best_radius, min_dist = 320, 240, 0, float('inf') 
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1500: 
            # GEOMETRY CHECK: Reject merged anomalies
            w, h = cv2.boundingRect(cnt)[2:]
            if 0.6 < (w / float(h)) < 1.4: 
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                # The target is the contour physically closest to the camera center
                dist = np.sqrt((x - 320)**2 + (y - 240)**2)
                if dist < min_dist:
                    min_dist = dist
                    best_x, best_y, best_radius = int(x), int(y), int(radius)
                    
    return best_x, best_y, best_radius

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. LOAD PREVIOUS STATE
    print("📂 Loading Ring Gear Coordinates from Step 1...")
    state_path = os.path.join(script_dir, "..", "config", "assembly_state.json")
    with open(state_path, "r") as f: state_data = json.load(f)
    ring_gear_x, ring_gear_y = state_data["ring_gear_x"], state_data["ring_gear_y"]

    rough_targets_path = os.path.join(script_dir, "..", "config", "red_gears_targets.json")

    T_cam2gripper = np.load(os.path.join(script_dir, "..", "config", "calibration.npy"))
    robot, gripper = Robot(ROBOT_IP), Gripper(ROBOT_IP)
    robot.recover_from_errors()
    robot.relative_dynamics_factor = 0.03

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(config)
    profile.get_device().first_depth_sensor().set_option(rs.option.emitter_enabled, 0)
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    try:
        print("\n👀 Robot Ready! Press 'y' to begin assembly, or 'q' to quit.")
        final_img = None
        while True:
            frames = pipeline.wait_for_frames()
            img = np.asanyarray(frames.get_color_frame().get_data())
            cv2.imshow("Global View", img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('y'):
                final_img = img.copy()
                cv2.destroyAllWindows()
                break
            elif key == ord('q'): return

        rough_targets = []
        if os.path.exists(rough_targets_path):
            print("\n📂 Found cached rough coordinates for the red gears!")
            with open(rough_targets_path, "r") as f:
                rough_targets = json.load(f)["rough_targets"]
        else:
            print("🧠 Asking Gemini to locate the 3 Red Gears (OpenCV Reality Filter Active)...")
            red_targets = ask_gemini_for_red_gears(final_img, num_gears=3)
            if not red_targets.get('red_gears') or len(red_targets['red_gears']) != 3:
                print(f"\n❌ Aborting: AI found {len(red_targets.get('red_gears', []))} gears.")
                return
            
            global_debug_img = final_img.copy()
            T_gripper2base = get_T_flange_to_base(robot.state)
            T_cam_in_base = T_gripper2base @ T_cam2gripper
            cam_pos, cam_rot = T_cam_in_base[:3, 3], T_cam_in_base[:3, :3]
            
            for idx, (ai_x, ai_y) in enumerate(red_targets['red_gears']):
                # Reality check: Snap hallucination to real gear
                px_x, px_y, found_contour = refine_gear_center(final_img, int(ai_x), int(ai_y))
                
                if found_contour is not None:
                    (cx, cy), radius = cv2.minEnclosingCircle(found_contour)
                    cv2.circle(global_debug_img, (int(cx), int(cy)), int(radius), (255, 0, 0), 2)
                
                # Draw AI Guess vs Reality
                cv2.drawMarker(global_debug_img, (int(ai_x), int(ai_y)), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
                cv2.circle(global_debug_img, (px_x, px_y), 5, (0, 255, 0), -1)
                cv2.putText(global_debug_img, f"Gear {idx+1}", (px_x + 10, px_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                ray_base = cam_rot @ np.array(rs.rs2_deproject_pixel_to_point(intr, [px_x, px_y], 1.0))
                t = (0.0 - cam_pos[2]) / ray_base[2]
                rough_point = cam_pos + (t * ray_base)
                rough_targets.append((rough_point[0], rough_point[1]))

            print("\n🛑 PAUSED (Global): Please review the Reality Filter vs AI Guesses.")
            while True:
                cv2.imshow("Global Vision Confirmation", global_debug_img)
                cv2.imwrite(os.path.join(script_dir, "step2_global_red_gears_ground_truth.jpg"), global_debug_img)
                print("💾 Saved step2_global_red_gears_ground_truth.jpg")
                key = cv2.waitKey(0) & 0xFF
                if key == ord('y'):
                    cv2.destroyWindow("Global Vision Confirmation")
                    break
                elif key == ord('q'): return

            os.makedirs(os.path.dirname(rough_targets_path), exist_ok=True)
            with open(rough_targets_path, "w") as f:
                json.dump({"rough_targets": rough_targets}, f)

        # --- PHASE 2: LOCAL REFINEMENT & ASSEMBLY ---
        angles = [20, 140, 260]
        
        for i, (rough_x, rough_y) in enumerate(rough_targets):
            print(f"\n🚁 Transiting to Hover Point above Gear {i+1}...")
            robot.recover_from_errors() 
            
            curr_T = get_T_flange_to_base(robot.state)
            curr_x, curr_y = curr_T[0,3], curr_T[1,3]
            
            robot.move(CartesianMotion(Affine([curr_x, curr_y, HOVER_Z], DOWN_QUAT)))
            mid_x, mid_y = (curr_x + rough_x) / 2, (curr_y + rough_y) / 2
            robot.move(CartesianMotion(Affine([mid_x, mid_y, HOVER_Z], DOWN_QUAT)))
            robot.move(CartesianMotion(Affine([rough_x, rough_y, HOVER_Z], DOWN_QUAT)))
            time.sleep(1.0) 
            
            frames = pipeline.wait_for_frames()
            local_color = np.asanyarray(frames.get_color_frame().get_data())
            local_depth_frame = frames.get_depth_frame()
            
            local_px_x, local_px_y, local_radius = find_gear_local(local_color)
            
            valid_depths = []
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    cy, cx = max(0, min(local_px_y + dy, 479)), max(0, min(local_px_x + dx, 639))
                    d = local_depth_frame.get_distance(cx, cy)
                    if d > 0.05: valid_depths.append(d)
            gear_depth = np.median(valid_depths) if valid_depths else HOVER_Z - GEAR_DROP_Z
            
            T_gripper2base_local = get_T_flange_to_base(robot.state) 
            T_cam_in_base_local = T_gripper2base_local @ T_cam2gripper
            point_cam = rs.rs2_deproject_pixel_to_point(intr, [local_px_x, local_px_y], gear_depth)
            pick_point = T_cam_in_base_local @ np.array([point_cam[0], point_cam[1], point_cam[2], 1.0])
            pick_x, pick_y = pick_point[0], pick_point[1]
            
            rad = np.deg2rad(angles[i])
            drop_x = ring_gear_x + (PLANETARY_RADIUS * np.cos(rad))
            drop_y = ring_gear_y + (PLANETARY_RADIUS * np.sin(rad))
            
            local_debug_img = local_color.copy()
            if local_radius > 0:
                cv2.circle(local_debug_img, (local_px_x, local_px_y), local_radius, (255, 0, 0), 2)
            cv2.circle(local_debug_img, (local_px_x, local_px_y), 5, (0, 255, 0), -1)
            cv2.putText(local_debug_img, "Target Peg", (local_px_x + 10, local_px_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.putText(local_debug_img, f"Raw Sensor Depth: {gear_depth:.3f}m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(local_debug_img, f"Target Plunge Z:  {HANDLE_PICK_Z:.3f}m", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(local_debug_img, f"Grasp Force:      20 N", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

            print(f"\n📋 FLIGHT PLAN FOR GEAR {i+1}:")
            print(f"  1. 🟦 MICRO ALIGN   : Shift to exact peg center")
            print(f"  2. 🟦 MICRO PLUNGE  : Straight down to Z={HANDLE_PICK_Z}m")
            print(f"  3. 🤏 GRASP         : Pinch peg at 20N force")
            print(f"  4. 🟦 MICRO LIFT    : Straight up to safety Z={HOVER_Z}m")
            print(f"  5. 🟦 MACRO TRANSIT : Waypoint sweep (via mid-table) to Drop Zone")
            print(f"  6. 🟦 MICRO INSERT  : Plunge & wiggle to mesh teeth at Z={GEAR_DROP_Z}m")
            
            print(f"\n🛑 PAUSED (Local Gear {i+1}): Please review the HUD and Flight Plan.")
            print("Press 'y' to execute, or 'q' to abort.")
            
            while True:
                cv2.imshow(f"HUD Confirmation - Gear {i+1}", local_debug_img)
                key = cv2.waitKey(0) & 0xFF
                if key == ord('y'):
                    print(f"\n✅ Flight Plan approved! Executing Gear {i+1}...")
                    cv2.imwrite(os.path.join(script_dir, f"step2_pick_confirmed_gear_{i+1}.jpg"), local_debug_img)
                    print(f"💾 Saved pick_confirmed_gear_{i+1}.jpg")
                    cv2.destroyWindow(f"HUD Confirmation - Gear {i+1}")
                    break
                elif key == ord('q'):
                    print("🛑 Aborted by user. Exiting safely.")
                    cv2.destroyAllWindows()
                    return

            # ====================================================
            # --- THE BREADCRUMB TRAJECTORY (ABSOLUTE MATH) ---
            # ====================================================

            gripper.move(0.10, 0.1) 
            
            robot.move(CartesianMotion(Affine([pick_x, pick_y, HOVER_Z], DOWN_QUAT)))
            
            try: robot.setCartesianImpedance([600.0, 600.0, 100.0, 50.0, 50.0, 10.0])
            except AttributeError: robot.set_cartesian_impedance([600.0, 600.0, 100.0, 50.0, 50.0, 10.0])
            
            robot.move(CartesianMotion(Affine([pick_x, pick_y, HANDLE_PICK_Z], DOWN_QUAT)))
            
            try: robot.setCartesianImpedance([3000.0, 3000.0, 3000.0, 300.0, 300.0, 300.0])
            except AttributeError: pass
            
            gripper.grasp(HANDLE_DIAMETER, 0.1, 20.0) 
            time.sleep(0.5)
            robot.recover_from_errors() 
            
            robot.move(CartesianMotion(Affine([pick_x, pick_y, HOVER_Z], DOWN_QUAT)))
            
            mid_drop_x, mid_drop_y = (pick_x + drop_x) / 2, (pick_y + drop_y) / 2
            
            # The gears are symmetric, so we just keep the wrist relaxed at DOWN_QUAT
            robot.move(CartesianMotion(Affine([mid_drop_x, mid_drop_y, HOVER_Z], DOWN_QUAT)))
            robot.move(CartesianMotion(Affine([drop_x, drop_y, HOVER_Z], DOWN_QUAT)))
            
            robot.move(CartesianMotion(Affine([drop_x, drop_y, GEAR_DROP_Z + 0.05], DOWN_QUAT)))
            
            try: robot.setCartesianImpedance([600.0, 600.0, 100.0, 50.0, 50.0, 10.0])
            except AttributeError: robot.set_cartesian_impedance([600.0, 600.0, 100.0, 50.0, 50.0, 10.0])
            
            # 🟦 ABSOLUTE CARTESIAN PLUNGE
            insert_motion = CartesianMotion(Affine([drop_x, drop_y, GEAR_DROP_Z], DOWN_QUAT))
            try:
                safe_stop = CartesianStopMotion()
                insert_motion.add_reaction(Reaction((Measure.FORCE_Z < -4.0) | (Measure.FORCE_Z > 4.0), safe_stop))
            except Exception:
                pass 
            robot.move(insert_motion)
            
            # 🟦 ABSOLUTE CARTESIAN WIGGLES (Centered around 0 yaw)
            wiggle = 0.08
            quat_l = Rotation.from_euler("xyz", [np.pi, 0, wiggle]).as_quat().tolist()
            quat_r = Rotation.from_euler("xyz", [np.pi, 0, -wiggle]).as_quat().tolist()
            
            for _ in range(2):
                robot.move(CartesianMotion(Affine([drop_x, drop_y, GEAR_DROP_Z], quat_l)))
                robot.move(CartesianMotion(Affine([drop_x, drop_y, GEAR_DROP_Z], quat_r)))
                robot.move(CartesianMotion(Affine([drop_x, drop_y, GEAR_DROP_Z], DOWN_QUAT)))
            
            gripper.move(0.10, 0.1)
            robot.recover_from_errors()
            
            robot.move(CartesianMotion(Affine([drop_x, drop_y, HOVER_Z], DOWN_QUAT)))
            
            try: robot.setCartesianImpedance([3000.0, 3000.0, 3000.0, 300.0, 300.0, 300.0])
            except AttributeError: pass
            
        print("\n🎉 STEP 2 COMPLETE: Equilateral triangle formed inside the ring gear!")

    except Exception as e:
        traceback.print_exc()
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()