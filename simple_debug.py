import cv2
import numpy as np
import time
import sounddevice as sd

class SimpleDebug:
    def __init__(self):
        self.cap = None
        self.reference_frame = None
        self.sample_rate = 44100
        self.amplitude = 0.0
        self.frequency = 250
        self.phase = 0
        
    def audio_callback(self, outdata, frames, time, status):
        """Generate vibration signal."""
        if self.amplitude > 0:
            dt = 1.0 / self.sample_rate
            t = np.arange(frames) * dt
            signal = self.amplitude * np.sin(2 * np.pi * self.frequency * t + self.phase)
            self.phase += frames * 2 * np.pi * self.frequency / self.sample_rate
            self.phase = self.phase % (2 * np.pi)
            outdata[:, 0] = signal
            outdata[:, 1] = signal
        else:
            outdata.fill(0)
    
    def run(self):
        # Initialize camera
        self.cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
        if not self.cap.isOpened():
            print("Camera failed!")
            return
        
        # Start audio stream
        stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=2,
            callback=self.audio_callback,
            blocksize=256
        )
        stream.start()
        
        print("=== SIMPLE GELSIGHT DEBUG ===")
        print("Controls:")
        print("  'c': Calibrate")
        print("  '1': Test vibration (low)")
        print("  '2': Test vibration (medium)")
        print("  '3': Test vibration (high)")
        print("  '0': Turn off vibration")
        print("  'q': Quit")
        print()
        
        cv2.namedWindow('Debug', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Debug', 1200, 800)
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                display_frame = frame.copy()
                
                # Create larger display frame
                h, w = display_frame.shape[:2]
                if w > 1000:  # If high resolution, resize for display
                    scale = 800.0 / w
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    display_frame = cv2.resize(display_frame, (new_w, new_h))
                    gray = cv2.resize(gray, (new_w, new_h))
                    if self.reference_frame is not None:
                        reference_resized = cv2.resize(self.reference_frame, (new_w, new_h))
                    else:
                        reference_resized = None
                else:
                    reference_resized = self.reference_frame
                
                # Add status text with larger font
                cv2.putText(display_frame, f"Amplitude: {self.amplitude:.3f}", (20, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                
                if reference_resized is not None:
                    # Calculate difference
                    diff = cv2.absdiff(gray, reference_resized)
                    
                    # Test multiple thresholds with larger text
                    for i, thresh in enumerate([8, 12, 16, 20]):  # Lower thresholds for fingers
                        contact_mask = diff > thresh
                        contact_area = np.sum(contact_mask) / contact_mask.size
                        
                        # Calculate roughness if contact detected
                        roughness = 0.0
                        if contact_area > 0.0005:  # Lower minimum contact area
                            roi_coords = np.where(contact_mask)
                            if len(roi_coords[0]) > 0:
                                y_min, y_max = np.min(roi_coords[0]), np.max(roi_coords[0])
                                x_min, x_max = np.min(roi_coords[1]), np.max(roi_coords[1])
                                
                                if (y_max - y_min) > 5 and (x_max - x_min) > 5:  # Smaller minimum region
                                    contact_roi = gray[y_min:y_max+1, x_min:x_max+1]
                                    
                                    if contact_roi.size > 50:  # Lower pixel requirement
                                        laplacian = cv2.Laplacian(contact_roi, cv2.CV_64F)
                                        roughness_raw = np.var(laplacian)
                                        # More sensitive roughness detection for smooth surfaces
                                        roughness = np.clip(roughness_raw / 2000.0, 0, 1)  # Lower divisor = more sensitive
                        
                        # More sensitive amplitude calculation for fingers
                        if contact_area > 0.0005:  # If any contact detected
                            base_amp = 0.4  # Higher base amplitude
                            contact_boost = min(1.0, contact_area * 200)  # More contact boost
                            roughness_boost = max(0.2, roughness)  # Minimum 0.2 even for smooth surfaces
                            expected_amp = base_amp * contact_boost * roughness_boost
                        else:
                            expected_amp = 0.0
                        
                        # Display metrics with MUCH larger text
                        y_pos = 100 + i * 60
                        color = (0, 255, 255) if contact_area > 0.0005 else (100, 100, 100)
                        
                        # Split into multiple lines for readability
                        cv2.putText(display_frame, f"Threshold {thresh}:", (20, y_pos), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                        cv2.putText(display_frame, f"  Area: {contact_area:.4f}", (20, y_pos + 25), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                        cv2.putText(display_frame, f"  Rough: {roughness:.3f}", (300, y_pos + 25), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                        cv2.putText(display_frame, f"  Vib: {expected_amp:.3f}", (500, y_pos + 25), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                        
                        # Auto-set amplitude based on best detection (threshold 12 for fingers)
                        if thresh == 12 and contact_area > 0.0005:
                            self.amplitude = expected_amp
                            self.frequency = 200 + roughness * 600  # 200-800 Hz range
                
                else:
                    cv2.putText(display_frame, "Press 'c' to calibrate!", (20, 150), 
                               cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 3)
                
                # Add instructions at bottom
                instructions = [
                    "Instructions:",
                    "1. Press 'c' to calibrate (no contact)",
                    "2. Touch sensor with finger",
                    "3. Press 1/2/3 to test manual vibration"
                ]
                
                for i, instruction in enumerate(instructions):
                    cv2.putText(display_frame, instruction, (20, display_frame.shape[0] - 120 + i * 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                cv2.imshow('Debug', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    self.reference_frame = gray.copy()
                    print("Calibrated! Reference frame set.")
                elif key == ord('1'):
                    self.amplitude = 0.2
                    print("Test vibration: LOW")
                elif key == ord('2'):
                    self.amplitude = 0.4
                    print("Test vibration: MEDIUM")
                elif key == ord('3'):
                    self.amplitude = 0.7
                    print("Test vibration: HIGH")
                elif key == ord('0'):
                    self.amplitude = 0.0
                    print("Vibration OFF")
                    
        finally:
            stream.stop()
            stream.close()
            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    debug = SimpleDebug()
    debug.run() 