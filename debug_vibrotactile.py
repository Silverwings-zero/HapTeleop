import cv2
import numpy as np
import time
import sounddevice as sd
from datetime import datetime

class DebugVibrotactileGelSight:
    """Debug version with verbose output and test features."""
    
    def __init__(self, camera_index=1):
        self.camera_index = camera_index
        self.cap = None
        self.reference_frame = None
        self.is_running = False
        
        # Audio parameters
        self.sample_rate = 44100
        self.test_frequency = 250  # Hz for test tone
        self.amplitude = 0.3
        self.phase = 0
        
        # Debug flags
        self.debug_contact = True
        self.debug_audio = True
        self.show_diff = True
        
    def initialize_camera(self):
        """Initialize camera with debug output."""
        print(f"[DEBUG] Attempting to connect to camera {self.camera_index}...")
        
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_MSMF)
        
        if not self.cap.isOpened():
            print(f"[DEBUG] Camera {self.camera_index} failed, trying camera 0...")
            self.cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
            
            if not self.cap.isOpened():
                print("[ERROR] No camera found!")
                return False
        
        # Get camera properties
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"[DEBUG] Camera initialized: {width}x{height} @ {fps:.1f} FPS")
        return True
    
    def audio_callback(self, outdata, frames, time, status):
        """Audio callback with debug output."""
        if status:
            print(f"[AUDIO DEBUG] Status: {status}")
        
        # Generate test tone
        dt = 1.0 / self.sample_rate
        t = np.arange(frames) * dt
        
        # Simple test tone
        signal = self.amplitude * np.sin(2 * np.pi * self.test_frequency * t + self.phase)
        self.phase += frames * 2 * np.pi * self.test_frequency / self.sample_rate
        self.phase = self.phase % (2 * np.pi)
        
        # Stereo output
        outdata[:, 0] = signal  # Left channel
        outdata[:, 1] = signal  # Right channel
    
    def test_audio(self):
        """Test if audio output is working."""
        print("\n[AUDIO TEST] Testing audio output...")
        print("You should hear a 250 Hz tone for 3 seconds...")
        
        try:
            # Start audio stream
            stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=2,
                callback=self.audio_callback,
                blocksize=256
            )
            
            with stream:
                time.sleep(3)  # Play for 3 seconds
            
            print("[AUDIO TEST] Audio test completed!")
            return True
            
        except Exception as e:
            print(f"[AUDIO ERROR] Audio test failed: {e}")
            return False
    
    def analyze_contact(self, frame):
        """Analyze contact with detailed debugging."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.reference_frame is None:
            print("[DEBUG] No reference frame set!")
            return None, None
        
        # Calculate difference
        diff = cv2.absdiff(gray, self.reference_frame)
        
        # Test different thresholds
        thresholds = [5, 10, 15, 20, 25, 30]
        
        if self.debug_contact:
            print("\n[CONTACT DEBUG] Threshold analysis:")
            for thresh in thresholds:
                contact_mask = diff > thresh
                contact_area = np.sum(contact_mask) / contact_mask.size
                if contact_area > 0:
                    mean_intensity = np.mean(diff[contact_mask])
                    print(f"  Threshold {thresh:2d}: Area={contact_area:.4f}, Intensity={mean_intensity:.1f}")
        
        # Use threshold of 15
        contact_mask = diff > 15
        contact_area = np.sum(contact_mask) / contact_mask.size
        
        return diff, contact_mask
    
    def calculate_roughness(self, frame, contact_mask):
        """Calculate roughness with debugging."""
        if not np.any(contact_mask):
            return 0.0
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Get contact region
        roi_coords = np.where(contact_mask)
        if len(roi_coords[0]) == 0:
            return 0.0
        
        y_min, y_max = np.min(roi_coords[0]), np.max(roi_coords[0])
        x_min, x_max = np.min(roi_coords[1]), np.max(roi_coords[1])
        
        # Extract contact region
        contact_roi = gray[y_min:y_max+1, x_min:x_max+1]
        
        if contact_roi.size < 100:
            if self.debug_contact:
                print(f"[DEBUG] Contact region too small: {contact_roi.size} pixels")
            return 0.0
        
        # Calculate roughness using Laplacian variance
        laplacian = cv2.Laplacian(contact_roi, cv2.CV_64F)
        roughness_raw = np.var(laplacian)
        roughness_normalized = np.clip(roughness_raw / 5000.0, 0, 1)
        
        if self.debug_contact:
            print(f"[ROUGHNESS DEBUG] Raw: {roughness_raw:.1f}, Normalized: {roughness_normalized:.3f}")
        
        return roughness_normalized
    
    def run_debug(self):
        """Run debug session."""
        print("=== GELSIGHT VIBROTACTILE DEBUG ===")
        print("This will help diagnose why you're not feeling vibration")
        print()
        
        # Test 1: Audio output
        print("TEST 1: Audio Output")
        audio_works = self.test_audio()
        if not audio_works:
            print("[ERROR] Audio system not working! Check speakers/headphones.")
            return
        
        input("Press Enter to continue to camera test...")
        
        # Test 2: Camera initialization
        print("\nTEST 2: Camera Initialization")
        if not self.initialize_camera():
            print("[ERROR] Camera initialization failed!")
            return
        
        # Test 3: Contact detection
        print("\nTEST 3: Contact Detection and Roughness Analysis")
        print("Press 'c' to calibrate (with nothing touching sensor)")
        print("Press 'q' to quit")
        print("Watch the debug output in console...")
        
        cv2.namedWindow('Debug GelSight', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Debug GelSight', 1200, 400)
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Create display
                display_height = 400
                display_width = 1200
                
                # Resize frame for display
                frame_resized = cv2.resize(frame, (400, 300))
                
                # Create combined display
                display = np.zeros((display_height, display_width, 3), dtype=np.uint8)
                display[50:350, 50:450] = frame_resized  # Original frame
                
                # Add labels
                cv2.putText(display, "Original", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Analyze contact if reference is set
                if self.reference_frame is not None:
                    diff, contact_mask = self.analyze_contact(frame)
                    
                    if diff is not None:
                        # Show difference image
                        diff_resized = cv2.resize(diff, (400, 300))
                        diff_colored = cv2.applyColorMap(diff_resized, cv2.COLORMAP_JET)
                        display[50:350, 500:900] = diff_colored
                        cv2.putText(display, "Difference", (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        
                        # Show contact mask
                        mask_resized = cv2.resize(contact_mask.astype(np.uint8) * 255, (400, 300))
                        mask_colored = cv2.applyColorMap(mask_resized, cv2.COLORMAP_HOT)
                        display[50:350, 950:1350] = mask_colored[:, :400]  # Crop to fit
                        cv2.putText(display, "Contact Mask", (950, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        
                        # Calculate metrics
                        contact_area = np.sum(contact_mask) / contact_mask.size
                        roughness = self.calculate_roughness(frame, contact_mask)
                        
                        # Display metrics
                        metrics_y = 370
                        cv2.putText(display, f"Contact Area: {contact_area:.4f}", (50, metrics_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(display, f"Roughness: {roughness:.3f}", (400, metrics_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Expected vibration amplitude
                        expected_amp = 0.3 * roughness * min(1.0, contact_area * 100)
                        cv2.putText(display, f"Expected Amplitude: {expected_amp:.3f}", (700, metrics_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                else:
                    cv2.putText(display, "Press 'c' to calibrate!", (500, 200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                
                cv2.imshow('Debug GelSight', display)
                
                # Handle input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    self.reference_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    print("[DEBUG] Reference frame set!")
                
                # Print debug info every 30 frames
                frame_count += 1
                if frame_count % 30 == 0 and self.reference_frame is not None:
                    self.debug_contact = True
                else:
                    self.debug_contact = False
                    
        except KeyboardInterrupt:
            pass
        
        finally:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            print("\n[DEBUG] Debug session completed!")

def main():
    debugger = DebugVibrotactileGelSight(camera_index=1)
    debugger.run_debug()

if __name__ == "__main__":
    main() 