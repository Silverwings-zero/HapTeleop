# GelSight Vibrotactile Feedback System

This proof of concept demonstrates real-time vibrotactile feedback based on texture analysis from a GelSight tactile sensor. The system converts tactile information into audio signals that can be perceived as haptic feedback through speakers or headphones.

## Features

### Texture Analysis
- **Surface Roughness**: Detected using Laplacian edge detection
- **Contact Area**: Percentage of sensor surface in contact
- **Contact Pressure**: Intensity of contact based on deformation
- **Gradient Magnitude**: Measures texture variation
- **Gradient Direction**: Directional texture patterns
- **Texture Frequency**: Dominant spatial frequency of surface patterns

### Vibrotactile Mapping
- **Roughness → Carrier Frequency**: Rough surfaces produce higher frequency vibrations (200-800 Hz)
- **Contact Pressure → Amplitude**: Stronger contact produces louder feedback
- **Texture Patterns → Modulation**: Complex textures create modulated vibrations (5-50 Hz)
- **Gradient Direction → Stereo Panning**: Directional texture creates left/right audio balance
- **Gradient Magnitude → High-frequency Component**: Sharp edges add harmonic content

## Installation

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure your GelSight sensor is connected and recognized as a camera device.

3. Connect speakers or headphones to your audio output.

## Usage

### Running the System

```bash
python vibrotactile_gelsight.py
```

### Controls

- **'c'**: Calibrate - Set the current frame as reference (do this with no contact first)
- **'q' or ESC**: Quit the application
- **'a'**: Toggle analysis display on/off
- **'+'/'='**: Increase vibrotactile sensitivity/amplitude
- **'-'**: Decrease vibrotactile sensitivity/amplitude

### Setup Procedure

1. **Initial Calibration**:
   - Start the system
   - Ensure nothing is touching the GelSight sensor
   - Press 'c' to set the reference frame
   - This establishes the baseline for contact detection

2. **Testing**:
   - Gently place your finger on the GelSight sensor
   - You should hear audio feedback through your speakers/headphones
   - Try different materials and textures to experience various feedback patterns

3. **Sensitivity Adjustment**:
   - Use '+' and '-' keys to adjust the feedback intensity
   - Start with lower sensitivity and increase as needed

## Technical Details

### Audio Output Specifications
- **Sample Rate**: 44.1 kHz (CD quality)
- **Channels**: Stereo (for directional feedback)
- **Buffer Size**: 1024 samples (~23ms latency)
- **Frequency Range**: 200-800 Hz (optimal for vibrotactile perception)

### Texture Analysis Algorithms

1. **Roughness Detection**:
   ```python
   laplacian = cv2.Laplacian(gray, cv2.CV_64F)
   roughness = np.std(laplacian) / 255.0
   ```

2. **Contact Detection**:
   ```python
   diff = cv2.absdiff(gray, reference_frame)
   contact_mask = diff > threshold
   ```

3. **Gradient Analysis**:
   ```python
   grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
   grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
   ```

### Signal Generation

The vibrotactile signal combines multiple components:

- **Carrier Wave**: Base frequency determined by surface roughness
- **Amplitude Modulation**: Based on contact pressure
- **Frequency Modulation**: Based on texture patterns
- **Stereo Panning**: Based on gradient direction

```python
# Example signal generation
carrier = np.sin(2 * π * carrier_freq * t)
modulation = 1 + 0.5 * np.sin(2 * π * modulation_freq * t)
signal = amplitude * carrier * modulation
```

## Configuration

### Camera Settings
- Default camera index: 1 (change in `main()` function if needed)
- Resolution: 640x480 (adjustable)
- Frame rate: 30 FPS

### Vibrotactile Parameters
You can modify these in the `VibrotactileGenerator` class:

```python
self.base_frequency = 250  # Hz
self.roughness_freq_range = (200, 800)  # Hz
self.modulation_freq_range = (5, 50)  # Hz
self.max_amplitude = 0.3  # 0-1 scale
```

## Applications

### Research Applications
- Telepresence systems
- Robotic teleoperation
- Texture recognition training
- Accessibility devices
- Human-computer interaction studies

### Extension Possibilities
- Multiple GelSight sensors for multi-point feedback
- Integration with VR/AR systems
- Machine learning for texture classification
- Haptic texture databases
- Real-time texture synthesis

## Troubleshooting

### Common Issues

1. **No Audio Output**:
   - Check that speakers/headphones are connected
   - Verify system audio settings
   - Try adjusting sensitivity with '+' key

2. **Camera Not Found**:
   - Check GelSight USB connection
   - Try changing camera index in the code
   - Verify camera permissions

3. **Poor Texture Detection**:
   - Ensure proper calibration with 'c' key
   - Check lighting conditions
   - Verify sensor surface is clean

4. **Audio Latency**:
   - The system uses a small buffer size for low latency
   - Latency should be ~23ms, imperceptible for most users

### Performance Optimization
- Close other camera applications
- Use a dedicated USB port for the GelSight sensor
- Ensure adequate system resources for real-time processing

## Safety Notes

- **Audio Volume**: Start with low volume to avoid hearing damage
- **Prolonged Use**: Take breaks to prevent audio fatigue
- **Sensor Care**: Handle the GelSight sensor carefully to avoid damage

## Future Enhancements

- [ ] Advanced texture classification using machine learning
- [ ] Multiple vibrotactile patterns for different materials
- [ ] Integration with force feedback devices
- [ ] Wireless audio transmission
- [ ] Mobile device compatibility
- [ ] Real-time texture synthesis and replay

## Technical Requirements

- Python 3.7+
- OpenCV 4.5+
- NumPy 1.20+
- SciPy 1.7+
- sounddevice 0.4+
- GelSight tactile sensor
- Audio output device (speakers/headphones)

---

For questions or issues, please refer to the main project documentation or create an issue in the repository. 