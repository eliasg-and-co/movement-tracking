# Movement Tracking Sonification System

Transform choreography into music. Every gesture controls a sound parameter.

**"Movement as art"**

**"Music that requires you were there."**

## Current Version: V1 (Proof of Concept)

Working system for dance and performance tracking. See `versions/v1_proof_of_concept/` for details.

## Quick Start

1. Install dependencies: `pip install ultralytics opencv-python python-osc`
2. Run extraction: `python versions/v1_proof_of_concept/extract_movement.py`
3. Playback to Ableton: `python versions/v1_proof_of_concept/playback_osc.py`

## Development Timeline

- **V1 (Jan 2026):** Proof of concept, 6 parameters, full-body averaging
- **V2 (In Progress):** Individual limb tracking, 14 parameters, confidence scoring
- **V3 (Planned):** Object tracking for basically anything else
- **V4 (Planned):** Movement tracking in camera for live installations/collaborations
  
See `CHANGELOG.md` for detailed version history.

## Contact

Built by Elias Goodman
