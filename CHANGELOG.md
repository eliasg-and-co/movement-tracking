# Changelog

## V1 - January 16, 2026

### Initial Release
- YOLO pose detection extracts 6 movement metrics
- OSC protocol sends data to Ableton Live
- Max4Live display device monitors incoming data
- Tested with FKA twigs x On Running video (62 seconds, 1557 frames)
- 86.7% detection rate

### Features
- Full-body center of mass tracking
- Limb extension calculation
- Velocity and acceleration metrics
- Real-time video playback synchronized with OSC
- 3-frame smoothing to reduce jitter

### Known Limitations
- Cannot track individual limbs separately
- No shot change detection
- No confidence scoring
- Struggles with occlusion
- Single-person only

### Technical Stack
- Python 3.8+
- Ultralytics YOLO v8
- OpenCV
- python-osc
- Ableton Live 11+
- Max for Live

### Test Results
- Video: 1280x720, 25 FPS
- Processing time: ~2 minutes for 62-second video
- Detection success: 1349/1557 frames (86.7%)
- OSC latency: <5ms
- Works for: Dance, yoga, martial arts, performance art
- Struggles with: Fast sports, cooking close-ups, skateboarding

## V2 - January 28, 2026

### Major Improvements
- **Individual limb tracking:** Separate tracking for right hand, left hand, right foot, left foot, torso, head
- **Optical flow integration:** Lucas-Kanade optical flow fills gaps when YOLO detection fails
- **Kalman filtering:** Velocity-based prediction for smooth trajectories
- **Confidence scoring:** Per-keypoint and overall confidence metrics
- **Shot change detection:** Automatic tracker reset on camera cuts

### Performance
- Detection rate: 89.6% (vs V1: 86.7%)
- Average confidence: 0.52-0.88 depending on shot quality
- 33 shot changes detected in test video
- Continuous tracking during brief occlusions

### New Parameters (14 total)
**Positions:**
- Right hand X/Y (ports 8020-8021)
- Left hand X/Y (ports 8022-8023)
- Right foot X/Y (ports 8024-8025)
- Left foot X/Y (ports 8026-8027)
- Torso X/Y (ports 8028-8029)
- Head Y (port 8030)

**Metrics:**
- Limb spread (port 8031)
- Body rotation (port 8032)
- Monitor port: 8040 (all parameters)

### Technical Details
- Lucas-Kanade optical flow (15x15 window, 2 pyramid levels)
- Velocity tracking with 0.9 decay (friction simulation)
- 5-frame confidence history per keypoint
- Hierarchical fallback: YOLO → Optical Flow → Kalman prediction

### Backwards Compatibility
- V1 continues to work on ports 8000-8010
- V2 uses separate port range (8020-8040)
- Both can run simultaneously for A/B testing
- Legacy data fields (center_x, center_y, extension) maintained in V2 JSON

### Known Limitations
- Optical flow requires previous frame context (resets on shot changes)
- Low confidence during extreme motion blur
- Single-person tracking only
- Memory usage ~2x higher than V1 due to optical flow

### Files
- `extract_movement_v2.py` - Enhanced extraction with optical flow
- `playback_osc_v2.py` - V2 playback on ports 8020-8040
- `movement_data_v2.json` - Output with individual limb positions
