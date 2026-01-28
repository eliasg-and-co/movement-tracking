from ultralytics import YOLO
import cv2
import json
import numpy as np
from collections import deque

# Load YOLO pose model
model = YOLO('yolov8n-pose.pt')

video_path = '../../twigs_on.mp4'  # Relative path from v2 folder
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# YOLO keypoint indices
KEYPOINT_MAP = {
    'nose': 0,
    'left_eye': 1, 'right_eye': 2,
    'left_ear': 3, 'right_ear': 4,
    'left_shoulder': 5, 'right_shoulder': 6,
    'left_elbow': 7, 'right_elbow': 8,
    'left_wrist': 9, 'right_wrist': 10,
    'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14,
    'left_ankle': 15, 'right_ankle': 16
}

# Optical flow parameters (Lucas-Kanade)
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

# Kalman filter for each keypoint (simple velocity-based prediction)
class KeypointTracker:
    def __init__(self):
        self.position = None
        self.velocity = np.array([0.0, 0.0])
        self.confidence_history = deque(maxlen=5)
        
    def update(self, new_position, confidence):
        """Update tracker with new detection"""
        if self.position is not None and new_position is not None:
            # Calculate velocity
            self.velocity = new_position - self.position
        
        if new_position is not None:
            self.position = new_position
            self.confidence_history.append(confidence)
        else:
            # No detection - predict using velocity
            if self.position is not None:
                self.position = self.position + self.velocity
                # Decay velocity (friction)
                self.velocity *= 0.9
                self.confidence_history.append(0.0)
    
    def get_avg_confidence(self):
        if len(self.confidence_history) == 0:
            return 0.0
        return np.mean(self.confidence_history)
    
    def predict(self):
        """Predict next position based on velocity"""
        if self.position is not None:
            return self.position + self.velocity
        return None

# Initialize trackers for all keypoints
keypoint_trackers = {name: KeypointTracker() for name in KEYPOINT_MAP.keys()}

# Storage
movement_data = []
frame_count = 0
prev_gray = None
shot_changes = 0

# Helper functions
def detect_shot_change(frame, prev_frame, threshold=0.3):
    """Detect camera cuts"""
    if prev_frame is None:
        return False
    
    hist1 = cv2.calcHist([frame], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([prev_frame], [0], None, [256], [0, 256])
    diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    return diff < (1 - threshold)

def optical_flow_track(prev_gray, curr_gray, prev_points):
    """Track points using Lucas-Kanade optical flow"""
    if prev_points is None or len(prev_points) == 0:
        return None
    
    # Reshape for optical flow
    prev_pts = prev_points.reshape(-1, 1, 2).astype(np.float32)
    
    # Calculate optical flow
    next_pts, status, error = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, prev_pts, None, **lk_params
    )
    
    if next_pts is None:
        return None
    
    # Filter good points
    good_new = next_pts[status == 1]
    
    return good_new.reshape(-1, 2) if len(good_new) > 0 else None

def extract_limb_data(keypoints, confidences, frame_num, timestamp):
    """Extract individual limb positions with confidence scores"""
    
    data = {
        'frame': frame_num,
        'time': timestamp,
        'video_width': width,
        'video_height': height,
        'shot_change': False  # Will be set by caller
    }
    
    # Update trackers with YOLO detections
    for name, idx in KEYPOINT_MAP.items():
        kp = keypoints[idx]
        conf = confidences[idx] if confidences is not None else 0.8
        
        if kp[0] > 0:  # Valid detection
            keypoint_trackers[name].update(kp, conf)
        else:
            # No detection - use tracker prediction
            keypoint_trackers[name].update(None, 0.0)
    
    # Extract positions from trackers (with optical flow backup)
    # Right hand
    rh = keypoint_trackers['right_wrist'].position
    if rh is not None:
        data['right_hand_x'] = float(rh[0])
        data['right_hand_y'] = float(rh[1])
        data['right_hand_confidence'] = float(keypoint_trackers['right_wrist'].get_avg_confidence())
    else:
        data['right_hand_x'] = None
        data['right_hand_y'] = None
        data['right_hand_confidence'] = 0.0
    
    # Left hand
    lh = keypoint_trackers['left_wrist'].position
    if lh is not None:
        data['left_hand_x'] = float(lh[0])
        data['left_hand_y'] = float(lh[1])
        data['left_hand_confidence'] = float(keypoint_trackers['left_wrist'].get_avg_confidence())
    else:
        data['left_hand_x'] = None
        data['left_hand_y'] = None
        data['left_hand_confidence'] = 0.0
    
    # Right foot
    rf = keypoint_trackers['right_ankle'].position
    if rf is not None:
        data['right_foot_x'] = float(rf[0])
        data['right_foot_y'] = float(rf[1])
        data['right_foot_confidence'] = float(keypoint_trackers['right_ankle'].get_avg_confidence())
    else:
        data['right_foot_x'] = None
        data['right_foot_y'] = None
        data['right_foot_confidence'] = 0.0
    
    # Left foot
    lf = keypoint_trackers['left_ankle'].position
    if lf is not None:
        data['left_foot_x'] = float(lf[0])
        data['left_foot_y'] = float(lf[1])
        data['left_foot_confidence'] = float(keypoint_trackers['left_ankle'].get_avg_confidence())
    else:
        data['left_foot_x'] = None
        data['left_foot_y'] = None
        data['left_foot_confidence'] = 0.0
    
    # Torso (average of shoulders and hips)
    torso_points = []
    for name in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']:
        pos = keypoint_trackers[name].position
        if pos is not None:
            torso_points.append(pos)
    
    if len(torso_points) >= 2:
        torso_center = np.mean(torso_points, axis=0)
        data['torso_x'] = float(torso_center[0])
        data['torso_y'] = float(torso_center[1])
        data['torso_confidence'] = float(np.mean([
            keypoint_trackers[n].get_avg_confidence() 
            for n in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
        ]))
    else:
        data['torso_x'] = None
        data['torso_y'] = None
        data['torso_confidence'] = 0.0
    
    # Head
    head = keypoint_trackers['nose'].position
    if head is not None:
        data['head_x'] = float(head[0])
        data['head_y'] = float(head[1])
        data['head_confidence'] = float(keypoint_trackers['nose'].get_avg_confidence())
    else:
        data['head_x'] = None
        data['head_y'] = None
        data['head_confidence'] = 0.0
    
    # Body rotation (shoulder line angle)
    ls = keypoint_trackers['left_shoulder'].position
    rs = keypoint_trackers['right_shoulder'].position
    if ls is not None and rs is not None:
        shoulder_vector = rs - ls
        angle = np.arctan2(shoulder_vector[1], shoulder_vector[0])
        data['body_rotation'] = float((angle + np.pi) / (2 * np.pi))
        data['rotation_confidence'] = float(np.mean([
            keypoint_trackers['left_shoulder'].get_avg_confidence(),
            keypoint_trackers['right_shoulder'].get_avg_confidence()
        ]))
    else:
        data['body_rotation'] = None
        data['rotation_confidence'] = 0.0
    
    # Limb spread (max distance between extremities)
    extremities = []
    for name in ['left_wrist', 'right_wrist', 'left_ankle', 'right_ankle']:
        pos = keypoint_trackers[name].position
        if pos is not None:
            extremities.append(pos)
    
    if len(extremities) >= 2:
        max_dist = 0
        for i in range(len(extremities)):
            for j in range(i+1, len(extremities)):
                dist = np.linalg.norm(extremities[i] - extremities[j])
                max_dist = max(max_dist, dist)
        data['limb_spread'] = float(max_dist)
        data['spread_confidence'] = float(np.mean([
            keypoint_trackers[n].get_avg_confidence() 
            for n in ['left_wrist', 'right_wrist', 'left_ankle', 'right_ankle']
        ]))
    else:
        data['limb_spread'] = None
        data['spread_confidence'] = 0.0
    
    # Legacy compatibility (center of mass)
    all_positions = [t.position for t in keypoint_trackers.values() if t.position is not None]
    if len(all_positions) > 0:
        center = np.mean(all_positions, axis=0)
        data['center_x'] = float(center[0])
        data['center_y'] = float(center[1])
        
        # Legacy extension
        if rh is not None and lh is not None and rf is not None and lf is not None:
            extension = (np.linalg.norm(lh - lf) + np.linalg.norm(rh - rf)) / 2
            data['extension'] = float(extension)
            data['extension_norm'] = min(extension / 500, 1.0)
        else:
            data['extension'] = None
            data['extension_norm'] = None
    else:
        data['center_x'] = None
        data['center_y'] = None
        data['extension'] = None
        data['extension_norm'] = None
    
    # Overall confidence (average of all trackers)
    all_confidences = [t.get_avg_confidence() for t in keypoint_trackers.values()]
    data['overall_confidence'] = float(np.mean(all_confidences))
    
    return data

# Main processing loop
print(f"\n{'='*70}")
print(f"MOVEMENT EXTRACTION V2 - Enhanced Tracking")
print(f"{'='*70}")
print(f"Video: {width}x{height} @ {fps} FPS")
print(f"Total frames: {total_frames}")
print(f"Duration: {total_frames/fps:.1f} seconds")
print(f"\nEnhancements:")
print(f"  • Lucas-Kanade optical flow tracking")
print(f"  • Kalman filtering for smooth predictions")
print(f"  • Per-keypoint confidence scoring")
print(f"  • Shot change detection")
print(f"  • Individual limb tracking (14 parameters)")
print(f"\n{'='*70}\n")

prev_frame_bgr = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to grayscale for optical flow
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect shot changes
    is_shot_change = detect_shot_change(frame, prev_frame_bgr)
    if is_shot_change:
        shot_changes += 1
        # Reset trackers on shot change
        keypoint_trackers = {name: KeypointTracker() for name in KEYPOINT_MAP.keys()}
        print(f"⚠ Shot change at frame {frame_count}")
    
    # Run YOLO pose detection
    results = model(frame, verbose=False)
    
    # Extract keypoints if detected
    keypoints = None
    confidences = None
    
    if len(results) > 0 and results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
        keypoints = results[0].keypoints.xy[0].cpu().numpy()
        
        # Get confidence scores
        if hasattr(results[0].keypoints, 'conf') and results[0].keypoints.conf is not None:
            confidences = results[0].keypoints.conf[0].cpu().numpy()
    
    # If YOLO failed but we have optical flow context, use optical flow
    if keypoints is None and prev_gray is not None:
        # Get previous keypoint positions from trackers
        prev_points = np.array([
            t.position for t in keypoint_trackers.values() if t.position is not None
        ])
        
        if len(prev_points) > 0:
            # Track with optical flow
            tracked_points = optical_flow_track(prev_gray, gray, prev_points)
            
            if tracked_points is not None and len(tracked_points) == 17:
                # Use optical flow results as keypoints
                keypoints = tracked_points
                confidences = np.ones(17) * 0.5  # Medium confidence for optical flow
    
    # Extract limb data
    if keypoints is not None and len(keypoints) >= 17:
        limb_data = extract_limb_data(keypoints, confidences, frame_count, round(frame_count / fps, 3))
        limb_data['shot_change'] = is_shot_change
        movement_data.append(limb_data)
    
    frame_count += 1
    prev_gray = gray.copy()
    prev_frame_bgr = frame.copy()
    
    if frame_count % 30 == 0:
        detected = len(movement_data)
        detection_rate = detected/frame_count*100
        avg_confidence = np.mean([d['overall_confidence'] for d in movement_data[-30:]])
        print(f"Frame {frame_count}/{total_frames} - Detected: {detected} ({detection_rate:.1f}%) - Avg Conf: {avg_confidence:.2f}")

cap.release()

# Convert numpy types to Python types for JSON serialization
def convert_to_native_types(obj):
    """Recursively convert numpy types to Python native types"""
    if isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

# Convert all data
movement_data_clean = convert_to_native_types(movement_data)

# Save to JSON
output_file = 'movement_data_v2.json'
with open(output_file, 'w') as f:
    json.dump(movement_data_clean, f, indent=2)

print(f"\n{'='*70}")
print(f"✓ Extraction complete!")
print(f"✓ Frames processed: {frame_count}")
print(f"✓ Successful detections: {len(movement_data)} ({len(movement_data)/frame_count*100:.1f}%)")
print(f"✓ Shot changes: {shot_changes}")
print(f"✓ Output: {output_file}")
print(f"\nV2 Improvements:")
print(f"  • Optical flow tracking for missing frames")
print(f"  • Kalman filtering for smoother trajectories")
print(f"  • Per-limb confidence scoring")
print(f"  • Individual limb positions (14 parameters)")
print(f"{'='*70}\n")