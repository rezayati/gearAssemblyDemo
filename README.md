
# gearAssemblyDemo

A purely Python-based robotic planetary gear assembly project using a **Franka Emika Research 3** robot and an **Intel RealSense D435** camera, powered by the **franky-control** library.

This project uses a **Hybrid AI + Computer Vision pipeline**:  
**Google's Gemini 2.5 Flash** provides high-level semantic understanding of the scene (finding the trays and gears), while strict **OpenCV morphological filters** enforce physical reality and millimeter precision.

---

# Hardware Requirements

- **Franka Emika Robot Arm** (IP: `192.168.15.33`)
- **Intel RealSense D435 Depth Camera** (Wrist-mounted)
- **Planetary Gear Set**
  - 1 Black Ring Gear
  - 3 Red Planetary Gears
  - 1 Yellow Sun Gear

---

# Setup & Installation

## Install Dependencies

Ensure your Python environment is active, then install the required packages:

```bash
pip install -r requirements.txt
````

---

## Configure AI Credentials

Create a `.env` file in the root directory of the project and add your **Google Gemini API key**:

```env
GEMINI_API_KEY=your_api_key_here
```

---

# Execution Pipeline

The assembly is broken down into modular steps.

> **Note:**
> If you physically move the assembly table or the ring gear, you must delete the JSON files in the `config/` folder to clear the robot's spatial memory and force a fresh scan.

---

# Step 0: Camera Calibration

Tape a **9×6 checkerboard** (30 mm squares) to the table.

This script generates the **Hand–Eye matrix**:

```
config/calibration.npy
```

Run:

```bash
python scripts/calibrate_camera.py
```

---

# Step 1: Locate the Ring Gear

Finds the large black housing using **AI approximation + Canny edge detection**, saving the coordinates to:

```
config/assembly_state.json
```

Run:

```bash
python scripts/step1_ring_gear.py
```

---

# Step 2: Assemble the Planetary Gears

1. Locates the **3 red gears using Gemini**
2. Filters out the **orange trays** using strict **HSV morphology**
3. Uses **Impedance Control** to gently plunge them into the ring gear in a **perfect equilateral triangle**

Run:

```bash
python scripts/step2_red_gears.py
```

---

# Step 3: Sun Gear (Coming Soon)

*To be implemented*

Dropping the **central yellow gear** to lock the planetary system together.

---

# Explainability & "Ground Truth" Receipts

To ensure total transparency of the vision system, the scripts generate **visual receipts** in the `scripts/` folder during execution.

### step1_ring_gear.jpg

Shows:

* The detected edge
* The mathematical center of the black housing

### step2_global_red_gears_ground_truth.jpg

Shows:

* **Gemini's initial guesses** (red crosses)
* **OpenCV's mathematically corrected reality** (blue circles / green dots)

### step2_pick_confirmed_gear_X.jpg

A **local hover snapshot** capturing the exact alignment of the gripper **right before the plunge**.

```
```
