# GelSight Data Streaming and Analysis

This repository contains scripts for streaming and analyzing data from GelSight tactile sensors.

## Files

- `gelsight_stream.py` - Main streaming application with real-time display and recording
- `gelsight_utils.py` - Utility functions for data processing and analysis
- `testGelsight.py` - Simple test script for capturing single frames
- `requirements.txt` - Python dependencies

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Connect your GelSight sensor to your computer via USB

## Usage

### Real-time Streaming (`gelsight_stream.py`)

Run the main streaming application:
```bash
python gelsight_stream.py
```

**Controls:**
- `q` or `ESC`: Quit the application
- `r`: Start/stop recording video
- `s`: Save current frame as image
- `c`: Calibrate (save current frame as reference)
- `SPACE`: Pause/unpause stream

**Features:**
- Real-time video display with FPS counter
- Recording to AVI format
- Frame saving with timestamps
- Reference frame calibration
- Automatic fallback to different camera indices

### Data Analysis (`gelsight_utils.py`)

Process recorded videos and extract tactile information:

```python
from gelsight_utils import batch_process_video, plot_force_timeline

# Process a recorded video
results = batch_process_video(
    video_path="gelsight_data/recording.avi",
    reference_frame_path="gelsight_data/reference.png"
)

# Plot force timeline
plot_force_timeline(results, save_path="force_analysis.png")
```

**Analysis Features:**
- Contact detection
- Force magnitude estimation
- Contact area calculation
- Batch processing of video files
- Data visualization and plotting

### Simple Frame Capture (`testGelsight.py`)

Basic script to capture and save a single frame:
```bash
python testGelsight.py
```

## Output Data

The streaming application creates a `gelsight_data/` directory containing:
- `gelsight_recording_YYYYMMDD_HHMMSS.avi` - Recorded videos
- `gelsight_frame_YYYYMMDD_HHMMSS.png` - Individual frames
- `gelsight_reference_YYYYMMDD_HHMMSS.png` - Reference frames for calibration

Analysis results are saved as:
- `analysis_results.txt` - CSV file with frame-by-frame analysis
- Force timeline plots (if requested)

## Troubleshooting

**Camera not detected:**
- Check USB connection to GelSight sensor
- Try different camera indices (modify `camera_index` parameter)
- Ensure no other applications are using the camera

**OpenCV errors:**
- Update OpenCV: `pip install --upgrade opencv-python`
- Check camera permissions in Windows settings

**No video display:**
- Ensure you're not running in a headless environment
- Check that display is properly configured

## Customization

You can modify camera settings in `gelsight_stream.py`:
```python
streamer = GelSightStreamer(
    camera_index=1,    # Change if GelSight is on different camera
    width=640,         # Adjust resolution
    height=480
)
```

For analysis parameters, modify thresholds in `gelsight_utils.py`:
```python
binary, contours = self.detect_contact(
    frame, 
    threshold=30,      # Sensitivity for contact detection
    min_area=100       # Minimum contact area
)
```

## API Reference

### GelSightStreamer Class
- `initialize_camera()` - Connect to camera
- `start_recording()` / `stop_recording()` - Video recording control
- `save_frame(frame)` - Save individual frames
- `calibrate(frame)` - Set reference frame

### GelSightProcessor Class
- `load_reference(path)` - Load reference frame
- `detect_contact(frame)` - Find contact areas
- `estimate_force_magnitude(frame)` - Estimate relative force
- `analyze_frame(frame)` - Comprehensive frame analysis

## Requirements

- Python 3.7+
- OpenCV 4.5+
- NumPy 1.20+
- Matplotlib 3.3+ (for analysis and plotting)
- Windows 10+ (for Microsoft Media Foundation support) 