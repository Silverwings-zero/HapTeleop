import cv2
import numpy as np
import time
import threading
import queue
from scipy import signal
from scipy.fft import fft, fftfreq
import sounddevice as sd
from datetime import datetime
import math

"""
GelSight Vibrotactile Feedback System

This script captures tactile data from a GelSight sensor, analyzes surface textures,
and provides real-time vibrotactile feedback through the audio output.

Features:
- Real-time texture analysis from GelSight sensor
- Multiple texture metrics (roughness, contact area, gradient magnitude)
- Configurable vibrotactile feedback patterns
- Audio output through system audio jack
- Real-time visualization of both video and audio signals

Texture-to-Vibration Mapping:
- Roughness → High-frequency vibrations (200-800 Hz)
- Contact pressure → Amplitude modulation
- Texture patterns → Low-frequency modulation (5-50 Hz)
- Directional gradients → Stereo panning effects

Requirements:
- opencv-python
- numpy
- scipy
- sounddevice
"""

class TextureAnalyzer:
    """Analyzes GelSight images for texture properties."""
    
    def __init__(self):
        self.reference_frame = None
        self.prev_frame = None
        self.contact_threshold = 12  # Lower threshold for fingers (was 15)
        self.min_contact_area = 0.0005  # Lower minimum for fingers (was 0.001)
        
    def set_reference(self, frame):
        """Set reference frame for difference calculations."""
        self.reference_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        print("Reference frame set - baseline established for contact detection")
        
    def analyze_frame(self, frame):
        """
        Analyze frame for texture properties.
        Returns dictionary with texture metrics.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Initialize metrics
        metrics = {
            'roughness': 0.0,
            'contact_area': 0.0,
            'gradient_magnitude': 0.0,
            'gradient_direction': 0.0,
            'texture_frequency': 0.0,
            'contact_pressure': 0.0,
            'has_contact': False
        }
        
        if self.reference_frame is not None:
            # Calculate difference from reference
            diff = cv2.absdiff(gray, self.reference_frame)
            
            # Contact detection with improved threshold for fingers
            contact_mask = diff > self.contact_threshold
            contact_area = np.sum(contact_mask) / contact_mask.size
            
            # Only register contact if above minimum area
            if contact_area > self.min_contact_area:
                metrics['has_contact'] = True
                metrics['contact_area'] = contact_area
                
                # Contact pressure (mean intensity of contact region)
                if np.sum(contact_mask) > 0:
                    contact_intensity = np.mean(diff[contact_mask])
                    metrics['contact_pressure'] = contact_intensity / 255.0
                    
                    # Only analyze texture properties when in contact
                    # Surface roughness using Laplacian on contact region
                    contact_region = gray[contact_mask]
                    if contact_region.size > 50:  # Lower requirement (was 100)
                        # Apply Laplacian to contact region
                        roi_coords = np.where(contact_mask)
                        if len(roi_coords[0]) > 0:
                            y_min, y_max = np.min(roi_coords[0]), np.max(roi_coords[0])
                            x_min, x_max = np.min(roi_coords[1]), np.max(roi_coords[1])
                            
                            # Extract contact region (smaller minimum size)
                            if (y_max - y_min) > 5 and (x_max - x_min) > 5:  # Was 10x10
                                contact_roi = gray[y_min:y_max+1, x_min:x_max+1]
                                
                                if contact_roi.size > 0:
                                    # More sensitive roughness calculation for fingers
                                    laplacian = cv2.Laplacian(contact_roi, cv2.CV_64F)
                                    roughness_raw = np.var(laplacian)
                                    
                                    # More sensitive normalization for smooth surfaces
                                    metrics['roughness'] = np.clip(roughness_raw / 2000.0, 0, 1)  # Was 5000
                                    
                                    # Gradient analysis on contact region
                                    grad_x = cv2.Sobel(contact_roi, cv2.CV_64F, 1, 0, ksize=3)
                                    grad_y = cv2.Sobel(contact_roi, cv2.CV_64F, 0, 1, ksize=3)
                                    
                                    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                                    metrics['gradient_magnitude'] = np.mean(gradient_magnitude) / 255.0
                                    
                                    # Dominant gradient direction
                                    angles = np.arctan2(grad_y, grad_x)
                                    metrics['gradient_direction'] = np.mean(angles)
                                    
                                    # Texture frequency analysis
                                    if contact_roi.size > 25:  # Lower requirement (was 50)
                                        try:
                                            f_transform = fft(contact_roi.flatten())
                                            freqs = fftfreq(len(f_transform))
                                            
                                            # Find dominant frequency (excluding DC component)
                                            power_spectrum = np.abs(f_transform[1:len(f_transform)//2])
                                            if len(power_spectrum) > 0:
                                                dominant_freq_idx = np.argmax(power_spectrum)
                                                metrics['texture_frequency'] = abs(freqs[dominant_freq_idx + 1])
                                        except:
                                            metrics['texture_frequency'] = 0.0
        
        return metrics

class VibrotactileGenerator:
    """Generates vibrotactile signals based on texture analysis."""
    
    def __init__(self, sample_rate=44100, buffer_size=256):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.time = 0
        self.is_running = False
        
        # Vibrotactile parameters
        self.base_frequency = 250  # Hz - base tactile frequency
        self.roughness_freq_range = (200, 800)  # Hz range for roughness
        self.modulation_freq_range = (5, 50)  # Hz range for texture modulation
        self.max_amplitude = 0.6  # Higher amplitude for better finger sensitivity
        
        # Current vibration parameters (for continuous generation)
        self.current_params = {
            'carrier_freq': self.base_frequency,
            'amplitude': 0.0,
            'modulation_freq': 10.0,
            'stereo_balance': 0.0,
            'gradient_component': 0.0
        }
        
        # Target parameters (from texture analysis)
        self.target_params = self.current_params.copy()
        
        # Smoothing parameters
        self.smoothing_factor = 0.95  # Higher = more smoothing (0.9-0.99)
        self.last_update_time = time.time()
        
        # Phase continuity
        self.carrier_phase = 0.0
        self.modulation_phase = 0.0
        
        # Startup muting to prevent initial vibration burst
        self.startup_time = time.time()
        self.mute_duration = 2.0  # Mute for 2 seconds at startup
        self.is_muted = True
        
    def texture_to_vibration(self, metrics):
        """
        Convert texture metrics to vibrotactile signal parameters.
        """
        # No vibration if no contact detected
        if not metrics['has_contact']:
            return {
                'carrier_freq': self.base_frequency,
                'amplitude': 0.0,
                'modulation_freq': 10.0,
                'stereo_balance': 0.0,
                'gradient_component': 0.0
            }
        
        # Map roughness to carrier frequency (primary feedback)
        roughness = np.clip(metrics['roughness'], 0, 1)
        carrier_freq = self.roughness_freq_range[0] + roughness * (
            self.roughness_freq_range[1] - self.roughness_freq_range[0])
        
        # Finger-optimized amplitude calculation
        contact_factor = np.clip(metrics['contact_pressure'], 0.1, 1.0)
        
        # More generous roughness boost for smooth surfaces like fingers
        roughness_boost = max(0.3, roughness)  # Minimum 0.3 even for smooth surfaces
        
        # Higher base amplitude for better finger sensitivity
        base_amplitude = self.max_amplitude * 1.5  # 50% increase
        
        # Amplitude is primarily driven by contact, with roughness modulation
        amplitude = base_amplitude * contact_factor * roughness_boost
        
        # Ensure we don't exceed maximum
        amplitude = min(amplitude, self.max_amplitude * 2.0)  # Allow up to 2x max
        
        # Map texture frequency to modulation frequency
        texture_freq = np.clip(metrics['texture_frequency'] * 1000, 0, 1)
        modulation_freq = self.modulation_freq_range[0] + texture_freq * (
            self.modulation_freq_range[1] - self.modulation_freq_range[0])
        
        # Map gradient direction to stereo balance (-1 to 1)
        stereo_balance = np.tanh(metrics['gradient_direction'])
        
        # Map gradient magnitude to additional high-freq component
        gradient_component = np.clip(metrics['gradient_magnitude'], 0, 1)
        
        return {
            'carrier_freq': carrier_freq,
            'amplitude': amplitude,
            'modulation_freq': modulation_freq,
            'stereo_balance': stereo_balance,
            'gradient_component': gradient_component
        }
    
    def smooth_parameters(self):
        """Smooth transition between current and target parameters."""
        # Check if we're still in startup mute period
        current_time = time.time()
        if current_time - self.startup_time < self.mute_duration:
            self.is_muted = True
            # Force amplitude to 0 during mute
            self.current_params['amplitude'] = 0.0
            self.target_params['amplitude'] = 0.0
            return
        else:
            self.is_muted = False
        
        # Check if texture data is stale (no updates for a while)
        time_since_update = current_time - self.last_update_time
        
        # If no updates for more than 100ms, gradually reduce amplitude
        if time_since_update > 0.1:  # 100ms timeout
            fade_factor = max(0.0, 1.0 - (time_since_update - 0.1) * 5.0)  # Fade over 200ms
            self.target_params['amplitude'] *= fade_factor
        
        # Smooth all parameters
        for key in self.current_params:
            self.current_params[key] = (
                self.smoothing_factor * self.current_params[key] + 
                (1 - self.smoothing_factor) * self.target_params[key]
            )
    
    def generate_continuous_signal(self, frames):
        """Generate continuous vibrotactile signal with phase continuity."""
        # Smooth parameter transitions
        self.smooth_parameters()
        
        # If muted, return silence
        if self.is_muted:
            return np.zeros((frames, 2), dtype=np.float32)
        
        # Generate time array for this buffer
        dt = 1.0 / self.sample_rate
        t = np.arange(frames) * dt
        
        # Generate carrier wave with continuous phase
        carrier_freq = self.current_params['carrier_freq']
        carrier_phase_increment = 2 * np.pi * carrier_freq * dt
        
        carrier_phases = self.carrier_phase + np.cumsum(np.full(frames, carrier_phase_increment))
        carrier = np.sin(carrier_phases)
        
        # Update carrier phase for continuity (keep bounded)
        self.carrier_phase = (self.carrier_phase + frames * carrier_phase_increment) % (2 * np.pi)
        
        # Generate modulation with continuous phase
        modulation_freq = self.current_params['modulation_freq']
        modulation_phase_increment = 2 * np.pi * modulation_freq * dt
        
        modulation_phases = self.modulation_phase + np.cumsum(np.full(frames, modulation_phase_increment))
        modulation = 1 + 0.5 * np.sin(modulation_phases)
        
        # Update modulation phase for continuity (keep bounded)
        self.modulation_phase = (self.modulation_phase + frames * modulation_phase_increment) % (2 * np.pi)
        
        # High-frequency component for gradient information
        gradient_component = self.current_params['gradient_component']
        gradient_signal = gradient_component * 0.3 * np.sin(carrier_phases * 2)
        
        # Combine signals
        amplitude = self.current_params['amplitude']
        mono_signal = amplitude * (carrier * modulation + gradient_signal)
        
        # Create stereo output with smooth panning
        stereo_balance = self.current_params['stereo_balance']
        left_gain = (1 - stereo_balance) / 2 + 0.5  # 0.5 to 1.0
        right_gain = (1 + stereo_balance) / 2 + 0.5  # 0.5 to 1.0
        
        left_channel = mono_signal * left_gain
        right_channel = mono_signal * right_gain
        
        # Combine into stereo array
        stereo_signal = np.column_stack([left_channel, right_channel])
        
        return stereo_signal.astype(np.float32)
    
    def audio_callback(self, outdata, frames, time, status):
        """Audio callback function for real-time audio output."""
        if status:
            print(f"Audio status: {status}")
        
        # Generate continuous signal directly in callback
        if self.is_running:
            try:
                signal_data = self.generate_continuous_signal(frames)
                outdata[:] = signal_data
            except Exception as e:
                print(f"Audio generation error: {e}")
                outdata.fill(0)
        else:
            outdata.fill(0)
    
    def start_audio_stream(self):
        """Start the audio output stream."""
        self.is_running = True
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=2,
            callback=self.audio_callback,
            blocksize=self.buffer_size
        )
        self.stream.start()
        print(f"Audio stream started: {self.sample_rate} Hz, buffer size: {self.buffer_size}")
        print(f"Muted for {self.mute_duration} seconds to prevent startup vibration...")
    
    def stop_audio_stream(self):
        """Stop the audio output stream."""
        self.is_running = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
    
    def update_signal(self, metrics):
        """Update the vibrotactile signal based on new texture metrics."""
        if not self.is_running:
            return
        
        # Convert texture metrics to target vibration parameters
        new_target_params = self.texture_to_vibration(metrics)
        
        # Update target parameters (will be smoothly interpolated)
        self.target_params.update(new_target_params)
        
        # Update timestamp
        self.last_update_time = time.time()
    
    def set_smoothing(self, smoothing_factor):
        """Set the parameter smoothing factor (0.0 = no smoothing, 0.99 = heavy smoothing)."""
        self.smoothing_factor = np.clip(smoothing_factor, 0.0, 0.99)

class VibrotactileGelSight:
    """Main class combining GelSight sensing with vibrotactile feedback."""
    
    def __init__(self, camera_index=1, width=640, height=480):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        
        # Initialize components
        self.texture_analyzer = TextureAnalyzer()
        self.vibrotactile_gen = VibrotactileGenerator()
        
        self.cap = None
        self.is_running = False
        self.show_analysis = True
        
    def initialize_camera(self):
        """Initialize camera connection."""
        print(f"Connecting to camera {self.camera_index}...")
        
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_MSMF)
        
        if not self.cap.isOpened():
            print(f"Failed to open camera {self.camera_index}. Trying camera 0...")
            self.cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
            
            if not self.cap.isOpened():
                print("Error: No camera found.")
                return False
        
        # Set properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Get actual properties
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.width = actual_width
        self.height = actual_height
        
        print(f"Camera initialized: {actual_width}x{actual_height}")
        return True
    
    def draw_analysis_overlay(self, frame, metrics):
        """Draw texture analysis results on frame."""
        # Create a copy for overlay
        display_frame = frame.copy()
        
        # Text properties - LARGER TEXT for better readability
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8  # Increased from 0.5
        thickness = 2  # Increased from 1
        
        # Background for text
        overlay = display_frame.copy()
        
        # Create semi-transparent panel at bottom
        panel_height = 220  # Increased from 180
        cv2.rectangle(overlay, (0, self.height - panel_height), (self.width, self.height), 
                     (0, 0, 0), -1)
        
        # Blend with original frame
        alpha = 0.7
        display_frame = cv2.addWeighted(display_frame, alpha, overlay, 1 - alpha, 0)
        
        # Contact status indicator - LARGER
        contact_color = (0, 255, 0) if metrics['has_contact'] else (0, 0, 255)
        contact_text = "CONTACT" if metrics['has_contact'] else "NO CONTACT"
        cv2.putText(display_frame, contact_text, (20, self.height - 180), 
                   font, 1.0, contact_color, 3)  # Increased size and thickness
        
        # Texture metrics - LARGER
        y_start = self.height - 150
        line_height = 30  # Increased from 20
        
        metrics_text = [
            f"Roughness: {metrics['roughness']:.3f}",
            f"Contact Area: {metrics['contact_area']:.4f}",
            f"Contact Pressure: {metrics['contact_pressure']:.3f}",
            f"Gradient Mag: {metrics['gradient_magnitude']:.3f}",
        ]
        
        for i, text in enumerate(metrics_text):
            color = (0, 255, 255) if metrics['has_contact'] else (100, 100, 100)
            cv2.putText(display_frame, text, (20, y_start + i * line_height),
                       font, font_scale, color, thickness)
        
        # Vibrotactile parameters on the right side - LARGER
        vib_params = self.vibrotactile_gen.texture_to_vibration(metrics)
        
        vib_text = [
            f"Carrier: {vib_params['carrier_freq']:.0f} Hz",
            f"Amplitude: {vib_params['amplitude']:.3f}",
            f"Modulation: {vib_params['modulation_freq']:.1f} Hz",
            f"Smoothing: {self.vibrotactile_gen.smoothing_factor:.2f}",
        ]
        
        # Add mute status
        if self.vibrotactile_gen.is_muted:
            vib_text.append("STATUS: MUTED")
        
        for i, text in enumerate(vib_text):
            color = (0, 255, 255) if not self.vibrotactile_gen.is_muted else (255, 0, 0)
            cv2.putText(display_frame, text, (self.width // 2, y_start + i * line_height),
                       font, font_scale, color, thickness)
        
        return display_frame
    
    def run(self):
        """Main execution loop."""
        if not self.initialize_camera():
            return
        
        # Start audio stream
        self.vibrotactile_gen.start_audio_stream()
        
        # Create display window
        cv2.namedWindow('GelSight Vibrotactile Feedback', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('GelSight Vibrotactile Feedback', 1000, 700)  # Larger window
        
        print("\nGelSight Vibrotactile Feedback System")
        print("Controls:")
        print("  'q' or ESC: Quit")
        print("  'c': Calibrate (set reference frame) - DO THIS FIRST!")
        print("  'a': Toggle analysis display")
        print("  '+'/'-': Increase/decrease vibrotactile sensitivity")
        print("  'u'/'j': Increase/decrease smoothing (for continuity)")
        print("  's': Save current frame")
        print("\nIMPORTANT: Press 'c' to calibrate with NO contact first!")
        print("Then place your finger on the GelSight sensor...")
        print("You should hear continuous vibrotactile feedback based on surface roughness!")
        print(f"Current smoothing: {self.vibrotactile_gen.smoothing_factor:.2f}")
        
        self.is_running = True
        frame_count = 0
        
        try:
            while self.is_running:
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    print("Error reading frame")
                    break
                
                # Analyze texture
                metrics = self.texture_analyzer.analyze_frame(frame)
                
                # Update vibrotactile signal
                self.vibrotactile_gen.update_signal(metrics)
                
                # Create display frame
                if self.show_analysis:
                    display_frame = self.draw_analysis_overlay(frame, metrics)
                else:
                    display_frame = frame
                
                cv2.imshow('GelSight Vibrotactile Feedback', display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # Quit
                    break
                elif key == ord('c'):  # Calibrate
                    self.texture_analyzer.set_reference(frame)
                elif key == ord('a'):  # Toggle analysis display
                    self.show_analysis = not self.show_analysis
                    print(f"Analysis display: {'ON' if self.show_analysis else 'OFF'}")
                elif key == ord('+') or key == ord('='):  # Increase sensitivity
                    self.vibrotactile_gen.max_amplitude = min(1.0, 
                        self.vibrotactile_gen.max_amplitude + 0.1)
                    print(f"Vibrotactile amplitude: {self.vibrotactile_gen.max_amplitude:.1f}")
                elif key == ord('-'):  # Decrease sensitivity
                    self.vibrotactile_gen.max_amplitude = max(0.1, 
                        self.vibrotactile_gen.max_amplitude - 0.1)
                    print(f"Vibrotactile amplitude: {self.vibrotactile_gen.max_amplitude:.1f}")
                elif key == ord('u'):  # Increase smoothing
                    new_smoothing = min(0.99, self.vibrotactile_gen.smoothing_factor + 0.05)
                    self.vibrotactile_gen.set_smoothing(new_smoothing)
                    print(f"Smoothing factor: {self.vibrotactile_gen.smoothing_factor:.2f}")
                elif key == ord('j'):  # Decrease smoothing
                    new_smoothing = max(0.0, self.vibrotactile_gen.smoothing_factor - 0.05)
                    self.vibrotactile_gen.set_smoothing(new_smoothing)
                    print(f"Smoothing factor: {self.vibrotactile_gen.smoothing_factor:.2f}")
                elif key == ord('s'):  # Save frame
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"gelsight_frame_{timestamp}.png"
                    if cv2.imwrite(filename, frame):
                        print(f"Frame saved: {filename}")
                    else:
                        print("Error: Could not save frame")
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\nSystem interrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        self.is_running = False
        
        if self.cap:
            self.cap.release()
        
        self.vibrotactile_gen.stop_audio_stream()
        cv2.destroyAllWindows()
        
        print("System cleaned up")

def main():
    """Main function."""
    print("GelSight Vibrotactile Feedback System")
    print("=====================================")
    print()
    print("This system analyzes tactile data from a GelSight sensor")
    print("and provides real-time vibrotactile feedback through audio output.")
    print()
    print("Make sure your speakers/headphones are connected!")
    print()
    
    # Create and run the system
    system = VibrotactileGelSight(
        camera_index=1,  # Adjust as needed
        width=640,
        height=480
    )
    
    system.run()

if __name__ == "__main__":
    main() 