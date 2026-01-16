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
