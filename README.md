# 🎨 Air Canvas

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-00A67E?style=for-the-badge&logo=google&logoColor=white)

**Draw in the air using your fingertips! ✨**

*A real-time hand tracking application that transforms your finger movements into digital art*

</div>

---

## 🌟 Features

| Feature | Description |
|---------|-------------|
| ✋ **Real-time Hand Tracking** | Uses MediaPipe for accurate hand landmark detection |
| 🎨 **Multi-Color Palette** | 9 vibrant colors to choose from |
| 🖌️ **Adjustable Brush Size** | Customize your drawing stroke thickness |
| 🧹 **Eraser Mode** | Three-finger gesture activates eraser |
| 💾 **Save Drawings** | Export your creations as PNG images |
| 🎯 **Smooth Drawing** | Exponential smoothing reduces jitter |
| 👁️ **Live Preview** | See your hand and drawing simultaneously |

---

## 🎮 Gesture Controls

| Gesture | Action |
|---------|--------|
| ☝️ **Index Finger Up** | Draw mode - move to paint |
| 🤏 **Pinch (Thumb + Index)** | Pause drawing / move without painting |
| ✋ **Open Palm (All 5 fingers)** | Clear the entire canvas |
| ✌️ **Two Fingers (Index + Middle)** | Change to next color |
| 🤟 **Three Fingers** | Eraser mode |

---

## ⌨️ Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Q` | Quit application |
| `C` | Clear canvas |
| `S` | Save drawing |
| `+` | Increase brush size |
| `-` | Decrease brush size |
| `1-5` | Quick color selection |

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Webcam
- **macOS**: Camera permissions for Terminal/Python

### Setup

1. **Clone or navigate to the project directory:**
   ```bash
   cd open-air-canvas
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate  # On Windows
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

3. **Start drawing by raising your index finger!**

4. **Use gestures to control the canvas:**
   - Raise index finger to draw
   - Pinch to pause
   - Open palm to clear
   - Two fingers to change color

5. **Press 'S' to save your masterpiece!**

---

## 🛠️ Technical Details

### Technologies Used

| Technology | Purpose |
|------------|---------|
| **Python** | Core programming language |
| **OpenCV** | Video capture, image processing, display |
| **MediaPipe** | Hand landmark detection and tracking |
| **NumPy** | Numerical operations and array handling |

### How It Works

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  📹 Webcam      │───▶│  🖐️ MediaPipe   │───▶│  🎯 Gesture     │
│  Video Capture  │    │  Hand Detection  │    │  Recognition    │
└─────────────────┘    └──────────────────┘    └────────┬────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  🖥️ Display    │◀───│  🔀 Blend        │◀───│  🖌️ Canvas     │
│  Combined View  │    │  Video + Canvas  │    │  Drawing        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

1. **Video Capture**: Webcam captures live video feed
2. **Hand Detection**: MediaPipe identifies hand landmarks (21 points)
3. **Gesture Recognition**: Finger positions analyzed to determine gesture
4. **Canvas Drawing**: Fingertip coordinates mapped to drawing on canvas
5. **Blending**: Canvas overlaid on video feed for live preview

### Key Classes & Functions

- `AirCanvas` - Main application class
  - `get_finger_states()` - Determines which fingers are raised
  - `detect_gesture()` - Identifies current hand gesture
  - `draw_on_canvas()` - Renders drawing on the canvas
  - `smooth_point()` - Applies smoothing to reduce jitter

---

## 🎨 Color Palette

| Color | Preview | Hotkey |
|-------|---------|--------|
| Blue | 🔵 | `1` |
| Green | 🟢 | `2` |
| Red | 🔴 | `3` |
| Yellow | 🟡 | `4` |
| Purple | 🟣 | `5` |
| Cyan | 🔷 | - |
| Orange | 🟠 | - |
| Pink | 🩷 | - |
| White | ⚪ | - |

---

## 📁 Project Structure

```
open-air-canvas/
├── air_canvas.py           # Main application
├── hand_landmarker.task    # MediaPipe hand detection model
├── requirements.txt        # Dependencies
└── README.md              # Documentation
```

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| Camera not detected | Grant camera permissions in System Preferences |
| "OpenCV: not authorized" | Enable camera access for Terminal |
| Laggy performance | Reduce resolution, close other applications |
| Hand not detected | Improve lighting, ensure hand is clearly visible |
| Jittery lines | Adjust `smoothing_factor` in the code |
| Model not found | Download hand_landmarker.task (see Installation) |

---

## 🌈 Tips for Best Results

1. **Lighting**: Ensure good, even lighting on your hand
2. **Background**: Use a contrasting background for better detection
3. **Distance**: Keep hand 1-3 feet from camera
4. **Steady movements**: Slower, deliberate movements create smoother lines
5. **Practice**: Gestures become natural with practice!

---

## 📝 License

This project is open source and available for educational purposes.

---

<div align="center">

**Made with ❤️ using Python, OpenCV & MediaPipe**

*Wave your hand and create magic! ✨*

</div>
