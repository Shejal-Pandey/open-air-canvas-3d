"""
🎨 QUANTUM AIR CANVAS - Sci-Fi Edition
=======================================
Simple & Intuitive Controls:

GESTURES:
☝️  1 Finger = Draw (close shape to create 3D object)
✌️  2 Fingers = Move + Rotate 3D shapes
👊 Fist = Clear Canvas

UI BUTTONS (Click with finger):
🎨 Color buttons - Select drawing color
➕➖ Brush size buttons - Adjust brush size  
🧽 Eraser button - Toggle eraser mode
"""

import cv2
import numpy as np
import time
import os
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import List, Optional
import random


class Particle:
    """Sci-fi particle effect."""
    def __init__(self, x, y, color):
        self.x, self.y = x, y
        self.vx = random.uniform(-2, 2)
        self.vy = random.uniform(-2, 2)
        self.life = 1.0
        self.decay = random.uniform(0.03, 0.06)
        self.color = color
        self.size = random.randint(2, 4)
    
    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= self.decay
        return self.life > 0
    
    def draw(self, canvas):
        if self.life > 0:
            color = tuple(int(c * self.life) for c in self.color)
            cv2.circle(canvas, (int(self.x), int(self.y)), self.size, color, -1)


class Shape3D:
    """A 3D shape with full rotation and translation."""
    
    def __init__(self, contour, color_data, center):
        self.original_points = contour.reshape(-1, 2).astype(np.float32)
        self.original_center = np.array([center[0], center[1]], dtype=np.float32)
        self.original_points = self.original_points - self.original_center
        self.position = np.array([center[0], center[1], 0.0], dtype=np.float32)
        self.rotation = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.depth = 30
        self.color = color_data
        self.is_selected = False
        self._create_3d_mesh()
    
    def _create_3d_mesh(self):
        n = len(self.original_points)
        self.front_face = np.zeros((n, 3), dtype=np.float32)
        self.front_face[:, :2] = self.original_points
        self.front_face[:, 2] = self.depth / 2
        self.back_face = np.zeros((n, 3), dtype=np.float32)
        self.back_face[:, :2] = self.original_points
        self.back_face[:, 2] = -self.depth / 2
    
    def rotate(self, dx, dy, dz):
        self.rotation[0] = (self.rotation[0] + dx) % (2 * np.pi)
        self.rotation[1] = (self.rotation[1] + dy) % (2 * np.pi)
        self.rotation[2] = (self.rotation[2] + dz) % (2 * np.pi)
    
    def move(self, dx, dy):
        self.position[0] += dx
        self.position[1] += dy
    
    def get_rotation_matrix(self):
        rx, ry, rz = self.rotation
        Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
        return (Rz @ Ry @ Rx).astype(np.float32)
    
    def project_point(self, p3d, fov=800):
        R = self.get_rotation_matrix()
        rotated = R @ p3d
        x = rotated[0] + self.position[0]
        y = rotated[1] + self.position[1]
        z = rotated[2] + self.position[2]
        scale = fov / (fov + z)
        return (int(x * scale + (1 - scale) * self.position[0]), 
                int(y * scale + (1 - scale) * self.position[1])), z, scale
    
    def get_projected_faces(self):
        faces = []
        front_t = [self.project_point(p) for p in self.front_face]
        back_t = [self.project_point(p) for p in self.back_face]
        n = len(front_t)
        
        # Side faces
        for i in range(n):
            ni = (i + 1) % n
            pts = [front_t[i][0], front_t[ni][0], back_t[ni][0], back_t[i][0]]
            avg_z = (front_t[i][1] + front_t[ni][1] + back_t[ni][1] + back_t[i][1]) / 4
            v1 = np.array([pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]])
            v2 = np.array([pts[3][0] - pts[0][0], pts[3][1] - pts[0][1]])
            normal = v1[0] * v2[1] - v1[1] * v2[0]
            faces.append({'type': 'side', 'points': pts, 'z': avg_z, 'normal': normal})
        
        # Front/back faces
        front_pts = [p[0] for p in front_t]
        back_pts = [p[0] for p in back_t]
        faces.append({'type': 'front', 'points': front_pts, 'z': sum(p[1] for p in front_t)/n, 'normal': -np.mean([p[2] for p in front_t])})
        faces.append({'type': 'back', 'points': back_pts, 'z': sum(p[1] for p in back_t)/n, 'normal': np.mean([p[2] for p in back_t])})
        return faces
    
    def contains_point(self, point):
        pts = [self.project_point(p)[0] for p in self.front_face]
        contour = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        return cv2.pointPolygonTest(contour, point, False) >= 0
    
    def get_bounds(self):
        pts = [self.project_point(p)[0] for p in self.front_face] + [self.project_point(p)[0] for p in self.back_face]
        pts = np.array(pts)
        return (pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max() - pts[:, 0].min(), pts[:, 1].max() - pts[:, 1].min())


class UIButton:
    """Clickable UI button."""
    def __init__(self, x, y, w, h, label, color, action):
        self.x, self.y = x, y
        self.w, self.h = w, h
        self.label = label
        self.color = color
        self.action = action
        self.hover = False
        self.active = False
    
    def contains(self, px, py):
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h
    
    def draw(self, canvas):
        # Glow effect when active or hover
        if self.active:
            cv2.rectangle(canvas, (self.x - 3, self.y - 3), (self.x + self.w + 3, self.y + self.h + 3), (0, 255, 255), 2)
        elif self.hover:
            cv2.rectangle(canvas, (self.x - 2, self.y - 2), (self.x + self.w + 2, self.y + self.h + 2), (100, 100, 100), 1)
        
        # Button background
        cv2.rectangle(canvas, (self.x, self.y), (self.x + self.w, self.y + self.h), self.color, -1)
        cv2.rectangle(canvas, (self.x, self.y), (self.x + self.w, self.y + self.h), (150, 150, 150), 1)
        
        # Label
        text_size = cv2.getTextSize(self.label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        tx = self.x + (self.w - text_size[0]) // 2
        ty = self.y + (self.h + text_size[1]) // 2
        cv2.putText(canvas, self.label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


class QuantumAirCanvas:
    """Futuristic Air Canvas with simple, intuitive controls."""
    
    def __init__(self):
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hand_landmarker.task')
        if not os.path.exists(model_path):
            raise FileNotFoundError("hand_landmarker.task not found!")
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(options)
        
        self.width, self.height = 1280, 720
        self.ui_height = 90
        self.sidebar_width = 80
        
        # Drawing color (white only)
        self.draw_color = (255, 255, 255)
        self.color_name = "WHITE"
        
        # Brush
        self.brush_size = 5
        self.min_brush = 2
        self.max_brush = 25
        
        # Eraser mode
        self.eraser_mode = False
        self.eraser_size = 30
        
        # Drawing
        self.drawing_layer = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.drawing_path = []
        self.min_stroke_points = 20
        
        # 3D Shapes
        self.shapes: List[Shape3D] = []
        self.selected_shape: Optional[Shape3D] = None
        
        # Smoothing
        self.prev_point = None
        self.smoothed_point = None
        self.smoothing = 0.55
        
        # Gesture tracking
        self.last_pos = None
        self.last_angle = None
        self.frame_timestamp = 0
        self.current_gesture = "STANDBY"
        self.gesture_cooldown = 0.5
        self.last_gesture_time = 0
        
        # Particles
        self.particles: List[Particle] = []
        
        # Camera preview
        self.preview_size = (160, 120)
        self.preview_pos = (20, self.height - 140)
        
        # Create UI Buttons
        self._create_buttons()
        
        # Hover/click tracking
        self.finger_on_button = None
        self.click_cooldown = 0
        
        print("⚡ QUANTUM AIR CANVAS initialized!")
    
    def _create_buttons(self):
        """Create all UI buttons."""
        self.buttons = []
        
        # Right sidebar buttons (x position)
        sidebar_x = self.width - self.sidebar_width
        btn_w, btn_h = 60, 40
        
        # Eraser button
        self.eraser_btn = UIButton(sidebar_x + 10, 110, btn_w, btn_h, "ERASE", (80, 80, 80), "toggle_eraser")
        self.buttons.append(self.eraser_btn)
        
        # Brush size buttons
        self.btn_plus = UIButton(sidebar_x + 10, 170, btn_w // 2 - 2, btn_h, "+", (60, 60, 60), "brush_plus")
        self.btn_minus = UIButton(sidebar_x + 10 + btn_w // 2 + 2, 170, btn_w // 2 - 2, btn_h, "-", (60, 60, 60), "brush_minus")
        self.buttons.append(self.btn_plus)
        self.buttons.append(self.btn_minus)
        
        # Clear button
        self.clear_btn = UIButton(sidebar_x + 10, 230, btn_w, btn_h, "CLEAR", (50, 50, 150), "clear")
        self.buttons.append(self.clear_btn)
        
        # Save button
        self.save_btn = UIButton(sidebar_x + 10, 290, btn_w, btn_h, "SAVE", (50, 100, 50), "save")
        self.buttons.append(self.save_btn)
        
    
    def add_particles(self, x, y, color, count=3):
        for _ in range(count):
            self.particles.append(Particle(x, y, color))
    
    def get_finger_states(self, landmarks, handedness):
        is_right = handedness == 'Right'
        thumb_up = (landmarks[4].x < landmarks[3].x) if is_right else (landmarks[4].x > landmarks[3].x)
        return [thumb_up, landmarks[8].y < landmarks[6].y, landmarks[12].y < landmarks[10].y,
                landmarks[16].y < landmarks[14].y, landmarks[20].y < landmarks[18].y]
    
    def get_position(self, landmarks, idx):
        return (int(landmarks[idx].x * self.width), int(landmarks[idx].y * self.height))
    
    def get_two_finger_center(self, landmarks):
        idx = self.get_position(landmarks, 8)
        mid = self.get_position(landmarks, 12)
        return ((idx[0] + mid[0]) // 2, (idx[1] + mid[1]) // 2)
    
    def get_two_finger_angle(self, landmarks):
        idx = self.get_position(landmarks, 8)
        mid = self.get_position(landmarks, 12)
        return math.atan2(mid[1] - idx[1], mid[0] - idx[0])
    
    def detect_gesture(self, landmarks, handedness):
        fingers = self.get_finger_states(landmarks, handedness)
        index_pos = self.get_position(landmarks, 8)
        palm_pos = self.get_position(landmarks, 0)
        count = sum(fingers)
        current_time = time.time()
        
        # Fist = Clear (held gesture)
        if count == 0:
            if current_time - self.last_gesture_time > self.gesture_cooldown:
                self.last_gesture_time = current_time
                return 'fist', palm_pos
            return 'wait', palm_pos
        
        # 2 fingers (index + middle) = Move & Rotate 3D
        if fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:
            center = self.get_two_finger_center(landmarks)
            return 'two_finger', center
        
        # 1 finger (index only) = Draw / Erase / Click buttons
        if fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]:
            return 'one_finger', index_pos
        
        return 'idle', index_pos
    
    def check_button_click(self, pos):
        """Check if finger is clicking a button."""
        for btn in self.buttons:
            if btn.contains(pos[0], pos[1]):
                return btn
        return None
    
    def handle_button_action(self, action):
        """Handle button click action."""
        if action == "toggle_eraser":
            self.eraser_mode = not self.eraser_mode
            self.eraser_btn.active = self.eraser_mode
            print(f"{'🧽 ERASER ON' if self.eraser_mode else '✏️ DRAW MODE'}")
        elif action == "brush_plus":
            self.brush_size = min(self.brush_size + 2, self.max_brush)
            self.eraser_size = min(self.eraser_size + 5, 60)
        elif action == "brush_minus":
            self.brush_size = max(self.brush_size - 2, self.min_brush)
            self.eraser_size = max(self.eraser_size - 5, 15)
        elif action == "clear":
            self.clear_canvas()
        elif action == "save":
            self.save_canvas()
    
    def smooth_point(self, new_point):
        if self.smoothed_point is None:
            self.smoothed_point = new_point
        else:
            self.smoothed_point = (
                int(self.smoothing * new_point[0] + (1 - self.smoothing) * self.smoothed_point[0]),
                int(self.smoothing * new_point[1] + (1 - self.smoothing) * self.smoothed_point[1])
            )
        return self.smoothed_point
    
    def draw_stroke(self, point):
        """Draw or erase with index finger."""
        # Don't draw in UI areas
        if point[1] < self.ui_height + 5 or point[0] > self.width - self.sidebar_width - 5:
            self.prev_point = None
            return
        
        smoothed = self.smooth_point(point)
        
        if self.eraser_mode:
            cv2.circle(self.drawing_layer, smoothed, self.eraser_size, (0, 0, 0), -1)
            self.add_particles(smoothed[0], smoothed[1], (100, 100, 100), 1)
        else:
            color = self.draw_color
            self.drawing_path.append(smoothed)
            if self.prev_point:
                # Glow effect
                cv2.line(self.drawing_layer, self.prev_point, smoothed, 
                        tuple(int(c * 0.4) for c in color), self.brush_size + 6)
                cv2.line(self.drawing_layer, self.prev_point, smoothed, color, self.brush_size)
            self.add_particles(smoothed[0], smoothed[1], color, 2)
            self._check_closed_shape()
        
        self.prev_point = smoothed
    
    def _check_closed_shape(self):
        """Check if drawing path forms a closed shape."""
        if len(self.drawing_path) < self.min_stroke_points:
            return
        
        start = np.array(self.drawing_path[0])
        end = np.array(self.drawing_path[-1])
        distance = np.linalg.norm(end - start)
        
        path_length = sum(
            np.linalg.norm(np.array(self.drawing_path[i]) - np.array(self.drawing_path[i-1]))
            for i in range(1, len(self.drawing_path))
        )
        
        # If start and end are close, and path is long enough, create 3D shape
        if distance < 50 and path_length > 150:
            self._create_3d_shape()
    
    def _create_3d_shape(self):
        """Convert closed path to 3D shape."""
        path_array = np.array(self.drawing_path, dtype=np.int32).reshape(-1, 1, 2)
        path_array = np.vstack([path_array, path_array[0:1]])
        
        epsilon = 0.015 * cv2.arcLength(path_array, True)
        simplified = cv2.approxPolyDP(path_array, epsilon, True)
        
        area = cv2.contourArea(simplified)
        if area < 1200:
            return
        
        M = cv2.moments(simplified)
        if M["m00"] == 0:
            return
        
        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        
        color = self.draw_color
        color_data = {
            'front': color,
            'side': tuple(int(c * 0.7) for c in color),
            'back': tuple(int(c * 0.4) for c in color),
            'glow': tuple(min(255, int(c * 1.2)) for c in color),
            'name': self.color_name
        }
        
        shape = Shape3D(simplified, color_data, (cx, cy))
        self.shapes.append(shape)
        
        # Clear the drawing path from the layer
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.drawContours(mask, [simplified], -1, 255, self.brush_size + 10)
        cv2.drawContours(mask, [simplified], -1, 255, -1)
        self.drawing_layer[mask > 0] = (0, 0, 0)
        
        # Add celebration particles
        for _ in range(20):
            self.particles.append(Particle(cx, cy, color))
        
        self.drawing_path = []
        self.prev_point = None
        self.smoothed_point = None
        print(f"✨ Created 3D {color_data['name']} shape!")
    
    def manipulate_shape(self, landmarks, point):
        """Move and rotate 3D shape with two fingers."""
        # Find or select shape
        if self.selected_shape is None:
            for shape in reversed(self.shapes):
                if shape.contains_point((float(point[0]), float(point[1]))):
                    shape.is_selected = True
                    self.selected_shape = shape
                    self.last_pos = point
                    self.last_angle = self.get_two_finger_angle(landmarks)
                    return
            return
        
        # Get current position and angle
        current_angle = self.get_two_finger_angle(landmarks)
        
        # Move shape (drag)
        if self.last_pos:
            dx = point[0] - self.last_pos[0]
            dy = point[1] - self.last_pos[1]
            self.selected_shape.move(dx, dy)
            
            # Rotate based on horizontal/vertical movement
            self.selected_shape.rotate(dy * 0.008, dx * 0.008, 0)
        
        # Z-axis rotation (finger twist)
        if self.last_angle is not None:
            delta_angle = current_angle - self.last_angle
            if delta_angle > math.pi:
                delta_angle -= 2 * math.pi
            elif delta_angle < -math.pi:
                delta_angle += 2 * math.pi
            self.selected_shape.rotate(0, 0, delta_angle * 0.6)
        
        self.last_pos = point
        self.last_angle = current_angle
    
    def release_selection(self):
        if self.selected_shape:
            self.selected_shape.is_selected = False
            self.selected_shape = None
        self.last_pos = None
        self.last_angle = None
    
    def clear_canvas(self):
        self.drawing_layer = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.shapes = []
        self.selected_shape = None
        self.drawing_path = []
        self.prev_point = None
        self.smoothed_point = None
        print("🗑️ Canvas cleared!")
    
    def save_canvas(self):
        filename = f"quantum_canvas_{time.strftime('%Y%m%d_%H%M%S')}.png"
        # Create save image (without UI)
        save_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for shape in sorted(self.shapes, key=lambda s: s.position[2], reverse=True):
            self.render_shape(save_img, shape)
        mask = np.any(self.drawing_layer > 0, axis=2)
        save_img[mask] = self.drawing_layer[mask]
        cv2.imwrite(filename, save_img)
        print(f"💾 Saved: {filename}")
    
    def render_shape(self, canvas, shape: Shape3D):
        """Render 3D shape with lighting."""
        faces = shape.get_projected_faces()
        faces.sort(key=lambda f: f['z'], reverse=True)
        color = shape.color
        
        for face in faces:
            pts = np.array(face['points'], dtype=np.int32)
            if len(pts) < 3:
                continue
            
            if face['type'] == 'front' and face.get('normal', 0) > 0:
                # Highlight on front face
                overlay = canvas.copy()
                cv2.fillPoly(overlay, [pts], color['glow'])
                cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)
                face_color = color['front']
            elif face['type'] == 'back' and face.get('normal', 0) > 0:
                face_color = color['back']
            elif face['type'] == 'side':
                t = np.clip((face.get('normal', 0) + 200) / 400, 0.3, 1.0)
                face_color = tuple(int(color['side'][i] * t + color['back'][i] * (1 - t)) for i in range(3))
            else:
                continue
            
            cv2.fillPoly(canvas, [pts], face_color)
            cv2.polylines(canvas, [pts], True, tuple(max(0, c - 30) for c in face_color), 1)
        
        # Selection indicator
        if shape.is_selected:
            x, y, w, h = [int(v) for v in shape.get_bounds()]
            pulse = (math.sin(time.time() * 6) + 1) / 2
            thick = int(2 + pulse * 2)
            cv2.rectangle(canvas, (x - 5, y - 5), (x + w + 5, y + h + 5), color['glow'], thick)
    
    def draw_neural_hand(self, canvas, landmarks, gesture):
        """Draw futuristic hand visualization."""
        connections = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),
                       (0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]
        positions = {i: self.get_position(landmarks, i) for i in range(21)}
        
        # Color based on mode
        if self.eraser_mode:
            base_color = (100, 100, 255)
        elif gesture == 'two_finger':
            base_color = (255, 150, 50)
        else:
            base_color = self.draw_color
        
        pulse = (math.sin(time.time() * 5) + 1) / 2
        
        # Connections
        for s, e in connections:
            p1, p2 = positions[s], positions[e]
            if p1[1] < self.ui_height or p2[1] < self.ui_height:
                continue
            cv2.line(canvas, p1, p2, tuple(int(c * 0.3) for c in base_color), 5)
            cv2.line(canvas, p1, p2, base_color, 2)
        
        # Nodes
        for idx, pos in positions.items():
            if pos[1] < self.ui_height:
                continue
            size = 8 if idx in [4, 8, 12, 16, 20] else 5
            cv2.circle(canvas, pos, size + 3, tuple(int(c * 0.4) for c in base_color), -1)
            cv2.circle(canvas, pos, size, base_color, -1)
            # Pulse rings on fingertips
            if idx in [4, 8, 12, 16, 20]:
                ring = int(12 + pulse * 5)
                cv2.circle(canvas, pos, ring, base_color, 1)
    
    def draw_ui(self, canvas):
        """Draw the UI with buttons."""
        # Top bar background
        for y in range(self.ui_height):
            alpha = 0.9 - (y / self.ui_height) * 0.5
            cv2.line(canvas, (0, y), (self.width, y), (int(25 * alpha),) * 3, 1)
        
        # Top bar line
        cv2.line(canvas, (0, self.ui_height), (self.width, self.ui_height), (0, 255, 255), 2)
        
        # Right sidebar background
        sidebar_x = self.width - self.sidebar_width
        cv2.rectangle(canvas, (sidebar_x, self.ui_height), (self.width, self.height), (20, 20, 20), -1)
        cv2.line(canvas, (sidebar_x, self.ui_height), (sidebar_x, self.height), (0, 255, 255), 2)
        
        # Title
        cv2.putText(canvas, "QUANTUM AIR CANVAS", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 80, 80), 3)
        cv2.putText(canvas, "QUANTUM AIR CANVAS", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        
        # Mode indicator
        mode = "ERASER" if self.eraser_mode else "DRAW"
        mode_color = (100, 100, 255) if self.eraser_mode else self.draw_color
        cv2.putText(canvas, f"MODE: {mode}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color, 1)
        
        # Gesture status
        cv2.putText(canvas, f"[{self.current_gesture}]", (150, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 1)
        
        # Draw all buttons
        for btn in self.buttons:
            btn.draw(canvas)
        
        
        # Brush size indicator in sidebar
        sidebar_x = self.width - self.sidebar_width
        cv2.putText(canvas, "BRUSH", (sidebar_x + 15, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
        
        size_to_show = self.eraser_size if self.eraser_mode else self.brush_size
        cv2.circle(canvas, (sidebar_x + 40, 250 + 80), min(size_to_show, 20), 
                  (100, 100, 255) if self.eraser_mode else self.draw_color, -1)
        cv2.putText(canvas, f"Size: {size_to_show}", (sidebar_x + 10, 250 + 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
        
        # Shapes count
        cv2.putText(canvas, f"Shapes: {len(self.shapes)}", (sidebar_x + 10, self.height - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 200), 1)
        
        # Instructions in sidebar
        instructions = [
            "GESTURES:",
            "1 Finger: Draw",
            "2 Fingers: Move",
            "& Rotate 3D",
            "Fist: Clear"
        ]
        for i, text in enumerate(instructions):
            cv2.putText(canvas, text, (sidebar_x + 5, 400 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 120), 1)
    
    def draw_camera_preview(self, canvas, camera_frame):
        x, y = self.preview_pos
        w, h = self.preview_size
        preview = cv2.resize(camera_frame, (w, h))
        cv2.rectangle(canvas, (x - 2, y - 2), (x + w + 2, y + h + 2), (0, 200, 200), 1)
        canvas[y:y + h, x:x + w] = preview
        cv2.putText(canvas, "LIVE", (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
    
    def draw_cursor(self, canvas, gesture, position):
        """Draw gesture cursor."""
        if not position or position[1] < self.ui_height:
            return
        
        pulse = (math.sin(time.time() * 8) + 1) / 2
        
        if gesture == 'one_finger':
            if self.eraser_mode:
                cv2.circle(canvas, position, self.eraser_size, (100, 100, 255), 2)
                cv2.circle(canvas, position, self.eraser_size + int(pulse * 4), (50, 50, 150), 1)
            else:
                color = self.draw_color
                cv2.circle(canvas, position, self.brush_size + 6, tuple(int(c * 0.5) for c in color), 2)
                cv2.circle(canvas, position, self.brush_size, color, -1)
                cv2.circle(canvas, position, self.brush_size + 12 + int(pulse * 4), color, 1)
        
        elif gesture == 'two_finger':
            cv2.circle(canvas, position, 30, (255, 150, 50), 2)
            # Rotation indicator
            for angle in [0, 90, 180, 270]:
                rad = math.radians(angle + time.time() * 50)
                ex = position[0] + int(25 * math.cos(rad))
                ey = position[1] + int(25 * math.sin(rad))
                cv2.circle(canvas, (ex, ey), 4, (255, 150, 50), -1)
    
    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            print("❌ Could not open webcam!")
            return
        
        cv2.namedWindow("Quantum Air Canvas", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Quantum Air Canvas", self.width, self.height)
        
        print("\n" + "=" * 55)
        print("⚡ QUANTUM AIR CANVAS - Sci-Fi Edition")
        print("=" * 55)
        print("\n✋ GESTURES:")
        print("   ☝️  1 Finger  → Draw (close shape for 3D)")
        print("   ✌️  2 Fingers → Move & Rotate 3D Shapes")
        print("   👊 Fist      → Clear Canvas")
        print("\n🖱️  BUTTONS (click with finger):")
        print("   🎨 Colors   → Select drawing color")
        print("   🧽 ERASE    → Toggle eraser mode")
        print("   ➕➖        → Adjust brush/eraser size")
        print("\n⌨️  KEYS: Q=Quit, C=Clear, S=Save, E=Eraser")
        print("=" * 55 + "\n")
        
        while True:
            ret, camera_frame = cap.read()
            if not ret:
                break
            
            camera_frame = cv2.flip(camera_frame, 1)
            display = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
            # Grid background
            draw_area_right = self.width - self.sidebar_width
            for x in range(0, draw_area_right, 40):
                cv2.line(display, (x, self.ui_height), (x, self.height), (12, 12, 12), 1)
            for y in range(self.ui_height, self.height, 40):
                cv2.line(display, (0, y), (draw_area_right, y), (12, 12, 12), 1)
            
            # Render 3D shapes
            for shape in sorted(self.shapes, key=lambda s: s.position[2], reverse=True):
                self.render_shape(display, shape)
            
            # Drawing layer
            mask = np.any(self.drawing_layer > 0, axis=2)
            display[mask] = self.drawing_layer[mask]
            
            # Update particles
            self.particles = [p for p in self.particles if p.update()]
            for p in self.particles:
                p.draw(display)
            
            # Hand detection
            rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            self.frame_timestamp += 33
            results = self.hand_landmarker.detect_for_video(mp_image, self.frame_timestamp)
            
            gesture, position = 'idle', None
            
            if results.hand_landmarks:
                landmarks = results.hand_landmarks[0]
                handedness = results.handedness[0][0].category_name if results.handedness else 'Right'
                gesture, position = self.detect_gesture(landmarks, handedness)
                
                self.draw_neural_hand(display, landmarks, gesture)
                
                # Update button hover states
                for btn in self.buttons:
                    btn.hover = btn.contains(position[0], position[1]) if position else False
                
                # Button click detection (with cooldown)
                if self.click_cooldown > 0:
                    self.click_cooldown -= 1
                
                if gesture == 'one_finger' and position:
                    clicked_btn = self.check_button_click(position)
                    if clicked_btn and self.click_cooldown == 0:
                        self.handle_button_action(clicked_btn.action)
                        self.click_cooldown = 15  # Cooldown frames
                        self.current_gesture = "CLICK"
                    elif position[0] < self.width - self.sidebar_width and position[1] > self.ui_height:
                        self.current_gesture = "ERASING" if self.eraser_mode else "DRAWING"
                        self.draw_stroke(position)
                        self.release_selection()
                    else:
                        self.current_gesture = "STANDBY"
                
                elif gesture == 'two_finger':
                    self.current_gesture = "MOVE/ROTATE"
                    self.manipulate_shape(landmarks, position)
                    self.prev_point = None
                    self.drawing_path = []
                
                elif gesture == 'fist':
                    self.current_gesture = "CLEAR"
                    self.clear_canvas()
                
                else:
                    self.current_gesture = "STANDBY"
                    self.prev_point = None
                    self.smoothed_point = None
                    if len(self.drawing_path) > 5:
                        self.drawing_path = []
                    self.release_selection()
            else:
                self.current_gesture = "NO HAND"
                self.prev_point = None
                self.smoothed_point = None
                self.drawing_path = []
                self.release_selection()
                for btn in self.buttons:
                    btn.hover = False
            
            # Draw UI
            self.draw_ui(display)
            self.draw_camera_preview(display, camera_frame)
            self.draw_cursor(display, gesture, position)
            
            cv2.imshow("Quantum Air Canvas", display)
            
            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.clear_canvas()
            elif key == ord('e'):
                self.eraser_mode = not self.eraser_mode
                self.eraser_btn.active = self.eraser_mode
            elif key == ord('s'):
                self.save_canvas()
            elif key in [ord('+'), ord('=')]:
                self.brush_size = min(self.brush_size + 2, self.max_brush)
                self.eraser_size = min(self.eraser_size + 5, 60)
            elif key == ord('-'):
                self.brush_size = max(self.brush_size - 2, self.min_brush)
                self.eraser_size = max(self.eraser_size - 5, 15)
            elif ord('1') <= key <= ord('8'):
                self.color_idx = key - ord('1')
                self.eraser_mode = False
                self.eraser_btn.active = False
        
        cap.release()
        cv2.destroyAllWindows()
        self.hand_landmarker.close()
        print("👋 Goodbye!")


def main():
    print("\n⚡ Starting QUANTUM AIR CANVAS...\n")
    canvas = QuantumAirCanvas()
    canvas.run()


if __name__ == "__main__":
    main()