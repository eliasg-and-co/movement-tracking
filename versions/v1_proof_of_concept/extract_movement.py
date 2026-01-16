from ultralytics import YOLO
import cv2
import json
import numpy as np

# Load YOLO pose model
model = YOLO('yolov8n-pose.pt')

video_path = 'twigs_on.mp4'
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_count = 0
movement_data = []

print(f"Processing {total_frames} frames at {fps} FPS...")
print(f"Video length: {total_frames/fps:.1f} seconds\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run pose detection
    results = model(frame, verbose=False)
    
    # Check if person detected and keypoints exist
    if len(results) > 0 and results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
        keypoints = results[0].keypoints.xy[0].cpu().numpy()
        
        if len(keypoints) >= 17:  # Need at least 17 keypoints
            # Calculate center of mass
            valid_points = keypoints[keypoints[:, 0] > 0]  # Filter out zero coordinates
            if len(valid_points) > 0:
                center_x = np.mean(valid_points[:, 0])
                center_y = np.mean(valid_points[:, 1])
                
                # Calculate limb extension
                left_wrist = keypoints[9]
                right_wrist = keypoints[10]
                left_ankle = keypoints[15]
                right_ankle = keypoints[16]
                
                # Only calculate if points are valid (not 0,0)
                if all(p[0] > 0 for p in [left_wrist, right_wrist, left_ankle, right_ankle]):
                    extension = (
                        np.linalg.norm(left_wrist - left_ankle) +
                        np.linalg.norm(right_wrist - right_ankle)
                    ) / 2
                    
                    # Normalize to 0-1 range (assuming max extension ~500 pixels)
                    extension_norm = min(extension / 500, 1.0)
                    
                    movement_data.append({
                        'frame': frame_count,
                        'time': round(frame_count / fps, 3),
                        'center_x': float(center_x),
                        'center_y': float(center_y),
                        'extension': float(extension),
                        'extension_norm': float(extension_norm)
                    })
    
    frame_count += 1
    if frame_count % 30 == 0:
        detected = len(movement_data)
        print(f"Processed {frame_count}/{total_frames} frames - Detected: {detected} ({detected/frame_count*100:.1f}%)")

cap.release()

# Save to JSON
with open('movement_data.json', 'w') as f:
    json.dump(movement_data, f, indent=2)

print(f"\n✓ Done! Extracted {len(movement_data)}/{frame_count} frames ({len(movement_data)/frame_count*100:.1f}% detection rate)")
print(f"✓ Saved to movement_data.json")