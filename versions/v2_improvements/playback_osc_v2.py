from pythonosc import udp_client
import json
import time
import cv2
import numpy as np

# Load V2 movement data
with open('movement_data_v2.json', 'r') as f:
    movement_data = json.load(f)

# V2 OSC clients - Individual limb ports (8020-8033)
client_rh_x = udp_client.SimpleUDPClient("127.0.0.1", 8020)
client_rh_y = udp_client.SimpleUDPClient("127.0.0.1", 8021)
client_lh_x = udp_client.SimpleUDPClient("127.0.0.1", 8022)
client_lh_y = udp_client.SimpleUDPClient("127.0.0.1", 8023)
client_rf_x = udp_client.SimpleUDPClient("127.0.0.1", 8024)
client_rf_y = udp_client.SimpleUDPClient("127.0.0.1", 8025)
client_lf_x = udp_client.SimpleUDPClient("127.0.0.1", 8026)
client_lf_y = udp_client.SimpleUDPClient("127.0.0.1", 8027)
client_torso_x = udp_client.SimpleUDPClient("127.0.0.1", 8028)
client_torso_y = udp_client.SimpleUDPClient("127.0.0.1", 8029)
client_head_y = udp_client.SimpleUDPClient("127.0.0.1", 8030)
client_limb_spread = udp_client.SimpleUDPClient("127.0.0.1", 8031)
client_body_rotation = udp_client.SimpleUDPClient("127.0.0.1", 8032)

# Monitor port (8040 - all V2 parameters)
monitor_client = udp_client.SimpleUDPClient("127.0.0.1", 8040)

# Calculate velocities
prev_data = None

for i, data in enumerate(movement_data):
    if prev_data:
        time_delta = data['time'] - prev_data['time']
        
        if time_delta > 0:
            # Right hand velocity
            if data['right_hand_x'] and prev_data['right_hand_x']:
                dx = data['right_hand_x'] - prev_data['right_hand_x']
                dy = data['right_hand_y'] - prev_data['right_hand_y']
                data['rh_velocity'] = min(np.sqrt(dx**2 + dy**2) / time_delta / 100, 1.0)
            else:
                data['rh_velocity'] = 0.0
            
            # Left hand velocity
            if data['left_hand_x'] and prev_data['left_hand_x']:
                dx = data['left_hand_x'] - prev_data['left_hand_x']
                dy = data['left_hand_y'] - prev_data['left_hand_y']
                data['lh_velocity'] = min(np.sqrt(dx**2 + dy**2) / time_delta / 100, 1.0)
            else:
                data['lh_velocity'] = 0.0
            
            # Torso velocity
            if data['torso_x'] and prev_data['torso_x']:
                dx = data['torso_x'] - prev_data['torso_x']
                dy = data['torso_y'] - prev_data['torso_y']
                data['torso_velocity'] = min(np.sqrt(dx**2 + dy**2) / time_delta / 100, 1.0)
            else:
                data['torso_velocity'] = 0.0
        else:
            data['rh_velocity'] = 0.0
            data['lh_velocity'] = 0.0
            data['torso_velocity'] = 0.0
    else:
        data['rh_velocity'] = 0.0
        data['lh_velocity'] = 0.0
        data['torso_velocity'] = 0.0
    
    prev_data = data

# Open video
cap = cv2.VideoCapture('../../twigs_on.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
frame_delay = 1.0 / fps
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"\n{'='*70}")
print(f"MOVEMENT TRACKING V2 PLAYBACK")
print(f"{'='*70}")
print(f"Video: {width}x{height} @ {fps} FPS")
print(f"\nV2 OSC Ports (Individual Limbs):")
print(f"  8020-8021: Right hand X/Y")
print(f"  8022-8023: Left hand X/Y")
print(f"  8024-8025: Right foot X/Y")
print(f"  8026-8027: Left foot X/Y")
print(f"  8028-8029: Torso X/Y")
print(f"  8030: Head Y")
print(f"  8031: Limb spread")
print(f"  8032: Body rotation")
print(f"  8040: Monitor (all parameters)")
print(f"\nTotal frames: {len(movement_data)}")
print(f"\nPress 'q' in video window to stop")
print(f"{'='*70}\n")

frame_idx = 0
data_idx = 0

while cap.isOpened():
    loop_start = time.time()
    
    ret, frame = cap.read()
    if not ret:
        break
    
    # Send OSC data
    if data_idx < len(movement_data):
        data = movement_data[data_idx]
        
        if data['frame'] == frame_idx:
            # Normalize positions
            rh_x = data['right_hand_x'] / width if data['right_hand_x'] else 0.5
            rh_y = data['right_hand_y'] / height if data['right_hand_y'] else 0.5
            lh_x = data['left_hand_x'] / width if data['left_hand_x'] else 0.5
            lh_y = data['left_hand_y'] / height if data['left_hand_y'] else 0.5
            rf_x = data['right_foot_x'] / width if data['right_foot_x'] else 0.5
            rf_y = 1 - (data['right_foot_y'] / height) if data['right_foot_y'] else 0.0  # Invert (0=ground)
            lf_x = data['left_foot_x'] / width if data['left_foot_x'] else 0.5
            lf_y = 1 - (data['left_foot_y'] / height) if data['left_foot_y'] else 0.0
            torso_x = data['torso_x'] / width if data['torso_x'] else 0.5
            torso_y = data['torso_y'] / height if data['torso_y'] else 0.5
            head_y = data['head_y'] / height if data['head_y'] else 0.3
            limb_spread = data['limb_spread'] / 500 if data['limb_spread'] else 0.5
            body_rot = data['body_rotation'] if data['body_rotation'] else 0.5
            
            # Send to individual ports
            client_rh_x.send_message("/v2/rh_x", rh_x)
            client_rh_y.send_message("/v2/rh_y", rh_y)
            client_lh_x.send_message("/v2/lh_x", lh_x)
            client_lh_y.send_message("/v2/lh_y", lh_y)
            client_rf_x.send_message("/v2/rf_x", rf_x)
            client_rf_y.send_message("/v2/rf_y", rf_y)
            client_lf_x.send_message("/v2/lf_x", lf_x)
            client_lf_y.send_message("/v2/lf_y", lf_y)
            client_torso_x.send_message("/v2/torso_x", torso_x)
            client_torso_y.send_message("/v2/torso_y", torso_y)
            client_head_y.send_message("/v2/head_y", head_y)
            client_limb_spread.send_message("/v2/limb_spread", limb_spread)
            client_body_rotation.send_message("/v2/body_rotation", body_rot)
            
            # Send to monitor
            monitor_client.send_message("/v2/rh_x", rh_x)
            monitor_client.send_message("/v2/rh_y", rh_y)
            monitor_client.send_message("/v2/lh_x", lh_x)
            monitor_client.send_message("/v2/lh_y", lh_y)
            monitor_client.send_message("/v2/rf_y", rf_y)
            monitor_client.send_message("/v2/lf_y", lf_y)
            monitor_client.send_message("/v2/torso_x", torso_x)
            monitor_client.send_message("/v2/torso_y", torso_y)
            monitor_client.send_message("/v2/head_y", head_y)
            monitor_client.send_message("/v2/limb_spread", limb_spread)
            monitor_client.send_message("/v2/body_rotation", body_rot)
            monitor_client.send_message("/v2/rh_velocity", data['rh_velocity'])
            monitor_client.send_message("/v2/lh_velocity", data['lh_velocity'])
            monitor_client.send_message("/v2/overall_confidence", data['overall_confidence'])
            
            data_idx += 1
    
    # Display with confidence overlay
    if data_idx > 0 and data_idx <= len(movement_data):
        conf = movement_data[data_idx-1]['overall_confidence']
        color = (0, 255, 0) if conf > 0.7 else (0, 165, 255) if conf > 0.4 else (0, 0, 255)
        
        cv2.putText(frame, f"Frame: {frame_idx} | Conf: {conf:.2f}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        if movement_data[data_idx-1].get('shot_change'):
            cv2.putText(frame, "SHOT CHANGE", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    
    cv2.imshow('Movement Tracking V2', frame)
    
    frame_idx += 1
    
    # Maintain framerate
    elapsed = time.time() - loop_start
    wait_time = max(1, int((frame_delay - elapsed) * 1000))
    
    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\n{'='*70}")
print(f"✓ V2 Playback complete")
print(f"✓ Sent {data_idx} frames")
print(f"{'='*70}\n")
