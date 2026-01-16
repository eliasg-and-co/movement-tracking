from pythonosc import udp_client
import json
import time
import cv2
import numpy as np

# Load movement data
with open('movement_data.json', 'r') as f:
    movement_data = json.load(f)

# OSC clients - individual ports for TouchOSC mapping (8000-8005)
client_center_x = udp_client.SimpleUDPClient("127.0.0.1", 8000)
client_center_y = udp_client.SimpleUDPClient("127.0.0.1", 8001)
client_extension = udp_client.SimpleUDPClient("127.0.0.1", 8002)
client_velocity = udp_client.SimpleUDPClient("127.0.0.1", 8003)
client_vertical_velocity = udp_client.SimpleUDPClient("127.0.0.1", 8004)
client_intensity = udp_client.SimpleUDPClient("127.0.0.1", 8005)

# Monitor port for display device (8010)
monitor_client = udp_client.SimpleUDPClient("127.0.0.1", 8010)

# Calculate enhanced metrics from raw data
enhanced_data = []
prev_data = None

for i, data in enumerate(movement_data):
    enhanced = data.copy()
    
    if prev_data:
        time_delta = data['time'] - prev_data['time']
        
        if time_delta > 0:
            # Overall velocity (speed of center point movement)
            dx = data['center_x'] - prev_data['center_x']
            dy = data['center_y'] - prev_data['center_y']
            distance = np.sqrt(dx**2 + dy**2)
            velocity = distance / time_delta
            
            # Vertical velocity specifically (jumps/drops)
            vertical_velocity = abs(dy) / time_delta
            
            # Extension change rate (how fast limbs open/close)
            extension_velocity = abs(data['extension'] - prev_data['extension']) / time_delta
            
            # Normalize velocities (cap at reasonable max)
            enhanced['velocity'] = min(velocity / 100, 1.0)  # Max ~100 pixels/sec
            enhanced['vertical_velocity'] = min(vertical_velocity / 50, 1.0)  # Max ~50 pixels/sec
            enhanced['intensity'] = min(extension_velocity / 50, 1.0)  # Extension speed
        else:
            enhanced['velocity'] = 0
            enhanced['vertical_velocity'] = 0
            enhanced['intensity'] = 0
    else:
        enhanced['velocity'] = 0
        enhanced['vertical_velocity'] = 0
        enhanced['intensity'] = 0
    
    enhanced_data.append(enhanced)
    prev_data = data

# Apply smoothing to reduce jitter (3-frame moving average)
def smooth(data_list, key, window=3):
    smoothed = []
    for i in range(len(data_list)):
        start = max(0, i - window // 2)
        end = min(len(data_list), i + window // 2 + 1)
        values = [data_list[j][key] for j in range(start, end)]
        smoothed.append(np.mean(values))
    return smoothed

# Smooth all parameters
for key in ['center_x', 'center_y', 'extension_norm', 'velocity', 'vertical_velocity', 'intensity']:
    smoothed_values = smooth(enhanced_data, key)
    for i, value in enumerate(smoothed_values):
        enhanced_data[i][f'{key}_smoothed'] = value

# Open video for display
cap = cv2.VideoCapture('twigs_on.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
frame_delay = 1.0 / fps
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

print(f"Enhanced Movement Playback")
print(f"========================")
print(f"Video: {fps} FPS, {width}x{height}")
print(f"\nOSC Ports:")
print(f"  Individual mapping (8000-8005):")
print(f"    8000: Horizontal position")
print(f"    8001: Vertical position")
print(f"    8002: Limb extension")
print(f"    8003: Movement velocity")
print(f"    8004: Vertical velocity")
print(f"    8005: Intensity")
print(f"  Monitor display: 8010 (all parameters)")
print(f"\nTotal frames: {len(enhanced_data)}")
print(f"\nPress 'q' in video window to stop\n")

frame_idx = 0
data_idx = 0

while cap.isOpened():
    start_time = time.time()
    
    ret, frame = cap.read()
    if not ret:
        break
    
    # Send OSC data if we have it for this frame
    if data_idx < len(enhanced_data) and enhanced_data[data_idx]['frame'] == frame_idx:
        data = enhanced_data[data_idx]
        
        # Use smoothed values
        center_x_norm = data['center_x'] / width
        center_y_norm = data['center_y'] / height
        extension_norm = data['extension_norm_smoothed']
        velocity_norm = data['velocity_smoothed']
        vertical_velocity_norm = data['vertical_velocity_smoothed']
        intensity_norm = data['intensity_smoothed']
        
        # Send to individual ports (for TouchOSC mapping)
        client_center_x.send_message("/movement/center_x", center_x_norm)
        client_center_y.send_message("/movement/center_y", center_y_norm)
        client_extension.send_message("/movement/extension", extension_norm)
        client_velocity.send_message("/movement/velocity", velocity_norm)
        client_vertical_velocity.send_message("/movement/vertical_velocity", vertical_velocity_norm)
        client_intensity.send_message("/movement/intensity", intensity_norm)
        
        # Send to monitor port (for display device)
        monitor_client.send_message("/movement/center_x", center_x_norm)
        monitor_client.send_message("/movement/center_y", center_y_norm)
        monitor_client.send_message("/movement/extension", extension_norm)
        monitor_client.send_message("/movement/velocity", velocity_norm)
        monitor_client.send_message("/movement/vertical_velocity", vertical_velocity_norm)
        monitor_client.send_message("/movement/intensity", intensity_norm)
        
        data_idx += 1
    
    # Display video with simple overlay
    overlay_text = f"Frame: {frame_idx} | Data points: {data_idx}"
    cv2.putText(frame, overlay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow('FKA twigs - Enhanced Movement Tracking', frame)
    
    frame_idx += 1
    
    # Maintain correct framerate
    elapsed = time.time() - start_time
    wait_time = max(1, int((frame_delay - elapsed) * 1000))
    
    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\n✓ Playback complete")
print(f"✓ Sent {data_idx} frames of enhanced movement data")