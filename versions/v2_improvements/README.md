# V2 - Individual Limb Tracking

**Date:** January 28, 2026  
**Status:** Production-ready for dance/choreography

## What's New in V2

### Individual Limb Tracking
Unlike V1's full-body averaging, V2 tracks each limb separately:
- Right hand, left hand (independent gesture control)
- Right foot, left foot (jump/step detection per leg)
- Torso (core movement)
- Head (vertical position for jumps/ducks)

### Optical Flow Integration
When YOLO fails to detect poses (motion blur, brief occlusion), Lucas-Kanade optical flow tracks keypoints from the previous frame. Result: smoother trajectories, fewer data gaps.

### Confidence Scoring
Every keypoint has a confidence score (0-1) indicating detection quality:
- **0.8-1.0:** Strong YOLO detection
- **0.5-0.7:** Optical flow tracking
- **0.0-0.4:** Kalman prediction (low confidence)

Use confidence to weight parameter mappings in Ableton.

## Performance vs V1

| Metric | V1 | V2 | Improvement |
|--------|----|----|-------------|
| Detection Rate | 86.7% | 89.6% | +2.9% |
| Parameters | 6 | 14 | +133% |
| Continuity | Gaps on occlusion | Optical flow fills gaps | Smoother |
| Confidence Data | None | Per-frame, per-limb | Better quality control |

## Usage

### 1. Extract Movement Data
```bash
python extract_movement_v2.py
```

Processes video and creates `movement_data_v2.json` with individual limb positions.

### 2. Playback to Ableton
```bash
python playback_osc_v2.py
```

Sends OSC data to ports 8020-8040. Video displays with confidence overlay (green = high, orange = medium, red = low).

### 3. Map in Ableton
Use TouchOSC devices or custom Max4Live devices on ports:
- **8020-8021:** Right hand X/Y
- **8022-8023:** Left hand X/Y  
- **8024-8025:** Right foot X/Y
- **8026-8027:** Left foot X/Y
- **8028-8029:** Torso X/Y
- **8030:** Head Y
- **8031:** Limb spread
- **8032:** Body rotation
- **8040:** Monitor (all parameters)

## When to Use V2 vs V1

**Use V2 for:**
- Dance choreography requiring individual gesture control
- Videos with motion blur or brief occlusions
- Projects needing high-quality confidence metadata
- Advanced mappings (right hand → melody, left hand → bass)

**Use V1 for:**
- Simple full-body tracking
- Lower CPU usage
- Learning the system
- Videos with minimal occlusion

## Technical Implementation

### Optical Flow
- Algorithm: Lucas-Kanade pyramid
- Window: 15x15 pixels
- Levels: 2 pyramid levels
- Criteria: 10 iterations or 0.03 epsilon

### Kalman Filtering
- State: Position + velocity per keypoint
- Prediction: `next_pos = current_pos + velocity`
- Friction: `velocity *= 0.9` (decay over time)
- Update: Velocity recalculated when YOLO re-detects

### Shot Change Detection
- Method: Histogram correlation
- Threshold: 0.7 correlation (below = shot change)
- Action: Reset all trackers on detection

## Known Issues

- Memory usage higher than V1 (~500MB for 60-second video)
- Optical flow computation adds ~30% processing time
- Low confidence during rapid spins (motion blur)
- Tracker reset on shot changes can cause brief parameter jumps

## Future Improvements

- Multi-person tracking with ID persistence
- GPU acceleration for optical flow
- Adaptive confidence thresholds
- Real-time camera input (not just video files)
