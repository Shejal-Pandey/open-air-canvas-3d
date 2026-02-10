# ⚡ QUANTUM AIR CANVAS — Sci-Fi Edition

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-00A67E?style=for-the-badge&logo=google&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243?style=for-the-badge&logo=numpy&logoColor=white)

**Draw in the air and create 3D objects with your fingertips! ✨**

*A futuristic hand tracking application with sci-fi UI, particle effects, and real-time 2D to 3D shape conversion*

</div>

---

## 🌟 Features

| Feature | Description |
|---------|-------------|
| ✋ **Real-time Hand Tracking** | Uses MediaPipe Hand Landmarker for accurate 21-point hand detection |
| 🖌️ **Air Drawing** | Draw in mid-air using your index finger with glow effects |
| � **2D → 3D Shape Conversion** | Close a drawn shape to automatically convert it into a 3D extruded object |
| 🎮 **3D Shape Manipulation** | Move and rotate 3D shapes using two-finger gestures |
| � **Eraser Mode** | Toggle eraser via on-screen button to erase parts of your drawing |
| ➕➖ **Adjustable Brush Size** | Increase/decrease brush and eraser size via UI buttons or keyboard |
| 💾 **Save Drawings** | Export your creations as PNG images |
| 🎯 **Smooth Drawing** | Exponential smoothing reduces jitter for cleaner strokes |
| ✨ **Particle Effects** | Sci-fi particle trails follow your drawing cursor |
| � **Live Camera Preview** | Small camera feed window in the bottom-left corner |
| 🖥️ **Sci-Fi HUD Interface** | Futuristic grid background, neon cyan accents, and neural hand visualization |
| 🖱️ **Clickable UI Buttons** | On-screen buttons in the sidebar — click them with your finger |

---

## 🎮 Gesture Controls

| Gesture | Fingers | Action |
|---------|---------|--------|
| ☝️ **One Finger** (Index only) | 1 | **Draw** — paint on the canvas (close a shape to create a 3D object) |
| ✌️ **Two Fingers** (Index + Middle) | 2 | **Move & Rotate** — drag and rotate selected 3D shapes |
| 👊 **Fist** (No fingers) | 0 | **Clear Canvas** — erases everything (with cooldown) |
| 🖐️ **Idle / Other** | 3+ | **Standby** — no action |

---

## 🖱️ UI Buttons (Right Sidebar)

Click any button by hovering your index finger over it:

| Button | Action |
|--------|--------|
| **ERASE** | Toggle eraser mode on/off |
| **+** | Increase brush/eraser size |
| **−** | Decrease brush/eraser size |
| **CLEAR** | Clear the entire canvas and all 3D shapes |
| **SAVE** | Save current canvas as a PNG image |

---

## ⌨️ Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Q` | Quit application |
| `C` | Clear canvas |
| `S` | Save drawing |
| `E` | Toggle eraser mode |
| `+` / `=` | Increase brush size |
| `-` | Decrease brush size |

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Webcam
- **macOS**: Camera permissions for Terminal/Python

### Setup

1. **Clone or navigate to the project directory:**
   ```bash
   cd open_air_canvas_draw
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On macOS/Linux
   # or
   .venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip3 install -r requirements.txt
   ```

4. **Download the hand landmarker model (if not present):**
   ```bash
   curl -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
   ```

---

## ⚠️ macOS Camera Permissions

On macOS, you need to grant camera access to Terminal (or your Python IDE):

1. Go to **System Preferences** → **Privacy & Security** → **Camera**
2. Enable camera access for **Terminal** (or your IDE like VS Code, PyCharm)
3. You may need to restart Terminal after granting permission

If you see "OpenCV: not authorized to capture video", this is the issue.

---

## 🎯 Usage

1. **Run the application:**
   ```bash
   python3 air_canvas.py
   ```

2. **Position yourself in front of the webcam**

3. **Raise your index finger to start drawing!**

4. **Create 3D shapes:**
   - Draw a closed shape (bring your stroke back to the starting point)
   - The shape will automatically convert into a rotatable 3D object!

5. **Manipulate 3D shapes:**
   - Hold up two fingers (index + middle)
   - Hover over a 3D shape to select it
   - Drag to move, twist fingers to rotate

6. **Use sidebar buttons** to toggle eraser, adjust brush size, clear, or save

7. **Press `Q` to quit**

---

## 🛠️ Technical Details

### Technologies Used

| Technology | Purpose |
|------------|---------|
| **Python** | Core programming language |
| **OpenCV** | Video capture, image processing, rendering, display |
| **MediaPipe** | Hand landmark detection (21-point model via Hand Landmarker Task API) |
| **NumPy** | Numerical operations, array handling, matrix math for 3D |

### Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  📹 Webcam      │───▶│  🖐️ MediaPipe   │───▶│  🎯 Gesture     │
│  Video Capture  │    │  Hand Landmarker │    │  Recognition    │
└─────────────────┘    └──────────────────┘    └────────┬────────┘
                                                         │
                       ┌─────────────────────────────────┼──────────────┐
                       │                                 │              │
                       ▼                                 ▼              ▼
              ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
              │  �️ 2D Drawing │    │  � 3D Shape     │    │  🧽 Eraser /    │
              │  Canvas Layer   │    │  Creation        │    │  UI Buttons     │
              └────────┬────────┘    └────────┬─────────┘    └────────┬────────┘
                       │                      │                       │
                       ▼                      ▼                       ▼
              ┌──────────────────────────────────────────────────────────────────┐
              │  🖥️ Sci-Fi HUD Renderer                                       │
              │  Grid bg + Neural hand + Particles + Camera preview + UI       │
              └──────────────────────────────────────────────────────────────────┘
```

1. **Video Capture**: Webcam captures live video at 1280×720
2. **Hand Detection**: MediaPipe Hand Landmarker identifies 21 hand landmarks
3. **Gesture Recognition**: Finger states analyzed to determine gesture (draw, move, fist, idle)
4. **2D Drawing**: Index fingertip coordinates are smoothed and rendered as brush strokes with glow
5. **3D Conversion**: Closed shapes are simplified, extruded into 3D meshes with front/back/side faces
6. **3D Manipulation**: Two-finger drag to move, finger twist to rotate (full 3-axis rotation)
7. **Rendering**: 3D shapes rendered with face-sorting and lighting; particles, HUD, and camera preview overlaid

### Key Classes & Methods

- **`Particle`** — Sci-fi particle effect with velocity, decay, and color
- **`Shape3D`** — 3D extruded shape with rotation matrices and perspective projection
  - `rotate(dx, dy, dz)` — Apply 3-axis rotation
  - `move(dx, dy)` — Translate shape position
  - `get_projected_faces()` — Project 3D faces to 2D for rendering
  - `contains_point(point)` — Hit-test for shape selection
- **`UIButton`** — Clickable on-screen button with hover/active states
- **`QuantumAirCanvas`** — Main application class
  - `detect_gesture()` — Classify hand gesture from finger states
  - `get_finger_states()` — Determine which fingers are raised
  - `draw_stroke()` — Render brush/eraser strokes on the canvas
  - `smooth_point()` — Apply exponential smoothing to reduce jitter
  - `_check_closed_shape()` — Detect when a drawn path forms a closed shape
  - `_create_3d_shape()` — Convert closed 2D path to extruded 3D shape
  - `manipulate_shape()` — Move and rotate 3D shapes with two-finger input
  - `render_shape()` — Render 3D shape with face sorting and lighting
  - `draw_neural_hand()` — Futuristic hand skeleton visualization
  - `draw_ui()` — Draw HUD, sidebar, buttons, and status indicators
  - `draw_cursor()` — Animated gesture cursor with pulse effects
  - `draw_camera_preview()` — Live camera feed in corner
  - `save_canvas()` — Export drawing + 3D shapes as PNG
  - `run()` — Main application loop

---

## 📁 Project Structure

```
open_air_canvas_draw/
├── air_canvas.py           # Main application (825 lines)
├── hand_landmarker.task    # MediaPipe hand detection model (~7.5 MB)
├── requirements.txt        # Python dependencies
├── README.md               # Documentation (this file)
└── *.png                   # Saved canvas images
```

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| Camera not detected | Grant camera permissions in System Preferences |
| "OpenCV: not authorized" | Enable camera access for Terminal |
| `hand_landmarker.task` not found | Download the model (see Installation step 4) |
| Laggy performance | Reduce resolution, close other applications |
| Hand not detected | Improve lighting, ensure hand is clearly visible |
| Jittery lines | Adjust `self.smoothing` value in the code (default: 0.55) |
| 3D shape not created | Draw a larger, fully closed shape (bring end point near start) |
| Fist clear too sensitive | Adjust `gesture_cooldown` value (default: 0.5s) |

---

## 🌈 Tips for Best Results

1. **Lighting**: Ensure good, even lighting on your hand
2. **Background**: Use a contrasting background for better detection
3. **Distance**: Keep hand 1–3 feet from camera
4. **Steady movements**: Slower, deliberate movements create smoother lines
5. **Close shapes**: To make 3D objects, bring your drawing stroke back to where you started
6. **Two-finger control**: Use index + middle finger to grab and rotate 3D shapes
7. **Practice**: Gestures become natural with practice!

---

## 📝 License

This project is open source and available for educational purposes.

---

<div align="center">

**Made with ❤️ using Python, OpenCV & MediaPipe**

*Wave your hand and create quantum art! ⚡✨*

</div>
