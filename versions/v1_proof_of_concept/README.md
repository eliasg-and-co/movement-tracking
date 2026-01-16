# V1 - Proof of Concept

**Date:** January 2026
**Status:** Working, production-ready for dance/choreography

## What It Does

Tracks full-body movement from video and sends 6 parameters to Ableton:
- Horizontal position (center of mass X)
- Vertical position (center of mass Y)
- Limb extension (how open the body is)
- Overall velocity (speed of movement)
- Vertical velocity (jump/crouch speed)
- Intensity (extension change rate)

## Limitations

- Averages all body parts (can't track individual hands separately)
- Struggles with occlusion and shot changes
- No confidence scoring
- 86.7% detection rate on test video

## Known Issues

- Parameter jumps during camera cuts
- Jittery data during fast movement
- Can't handle multiple people in frame

## What Works Great

- Dance performances (FKA twigs demo)
- Yoga/martial arts
- Solo performance art
- Well-lit, full-body shots

## Files

- `extract_movement.py` - Video → JSON data
- `playback_osc.py` - JSON → OSC to Ableton
- `MOVEMENT.amxd` - Max4Live display device

## Usage
```bash
python extract_movement.py  # Creates movement_data.json
python playback_osc.py      # Plays video + sends OSC
```

In Ableton: Add MOVEMENT.amxd to see incoming data, use TouchOSC devices on ports 8000-8005 to map to parameters.
