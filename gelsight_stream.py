import cv2
import numpy as np
import time
import os
from datetime import datetime

"""
GelSight Data Streaming Script

This script streams live data from a GelSight tactile sensor.
Features:
- Real-time video display
- Frame rate monitoring
- Data recording capabilities
- Keyboard controls for interaction

Controls:
- 'q' or ESC: Quit
- 'r': Start/stop recording
- 's': Save current frame
- 'c': Calibrate (save current frame as reference)
- SPACE: Pause/unpause stream

Requirements:
- opencv-python
- numpy
"""

class GelSightStreamer:
    def __init__(self, camera_index=1, width=640, height=480, display_scale=0.8):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.display_scale = display_scale  # Scale factor for display window
        self.cap = None
        self.is_recording = False
        self.is_paused = False
        self.frame_count = 0
        self.start_time = time.time()
        self.reference_frame = None
        self.video_writer = None
        
        # Create output directory if it doesn't exist
        self.output_dir = "gelsight_data"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def initialize_camera(self):
        """Initialize the camera connection."""
        print(f"Attempting to connect to camera {self.camera_index}...")
        
        # Try the specified camera index first
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_MSMF)
        
        if not self.cap.isOpened():
            print(f"Failed to open camera {self.camera_index}. Trying camera 0...")
            self.cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
            
            if not self.cap.isOpened():
                print("Error: No camera found. Please check your GelSight connection.")
                return False
        
        # Set camera properties (try to set, but use actual values)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Get actual camera properties and update our settings
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Update our dimensions to match actual camera
        self.width = actual_width
        self.height = actual_height
        
        print(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")
        return True
    
    def start_recording(self):
        """Start recording video."""
        if self.is_recording:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = os.path.join(self.output_dir, f"gelsight_recording_{timestamp}.avi")
        
        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter(video_filename, fourcc, 20.0, (self.width, self.height))
        
        if self.video_writer.isOpened():
            self.is_recording = True
            print(f"Started recording: {video_filename}")
        else:
            print("Error: Could not start video recording")
    
    def stop_recording(self):
        """Stop recording video."""
        if not self.is_recording:
            return
            
        self.is_recording = False
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        print("Recording stopped")
    
    def save_frame(self, frame):
        """Save current frame as image."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = os.path.join(self.output_dir, f"gelsight_frame_{timestamp}.png")
        
        if cv2.imwrite(filename, frame):
            print(f"Frame saved: {filename}")
        else:
            print("Error: Could not save frame")
    
    def calibrate(self, frame):
        """Save current frame as reference for difference calculations."""
        self.reference_frame = frame.copy()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"gelsight_reference_{timestamp}.png")
        
        if cv2.imwrite(filename, frame):
            print(f"Reference frame saved: {filename}")
        else:
            print("Error: Could not save reference frame")
    
    def process_frame(self, frame):
        """Process frame and add overlays."""
        processed_frame = frame.copy()
        
        # Calculate FPS
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        # Calculate text scale based on frame size
        text_scale = min(self.width, self.height) / 800.0
        text_scale = max(0.3, min(1.0, text_scale))  # Clamp between 0.3 and 1.0
        
        # Add text overlays
        cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, int(30 * text_scale + 15)), 
                   cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 255, 0), max(1, int(2 * text_scale)))
        
        cv2.putText(processed_frame, f"Frame: {self.frame_count}", (10, int(60 * text_scale + 15)), 
                   cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 255, 0), max(1, int(2 * text_scale)))
        
        cv2.putText(processed_frame, f"Size: {self.width}x{self.height}", (10, int(90 * text_scale + 15)), 
                   cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 255, 0), max(1, int(2 * text_scale)))
        
        # Recording indicator
        if self.is_recording:
            circle_radius = max(5, int(10 * text_scale))
            cv2.circle(processed_frame, (self.width - int(30 * text_scale), int(30 * text_scale)), 
                      circle_radius, (0, 0, 255), -1)
            cv2.putText(processed_frame, "REC", (self.width - int(80 * text_scale), int(35 * text_scale)), 
                       cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 0, 255), max(1, int(2 * text_scale)))
        
        # Paused indicator
        if self.is_paused:
            cv2.putText(processed_frame, "PAUSED", (self.width//2 - int(80 * text_scale), self.height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, text_scale * 1.5, (0, 255, 255), max(1, int(3 * text_scale)))
        
        return processed_frame
    
    def stream(self):
        """Main streaming loop."""
        if not self.initialize_camera():
            return
        
        # Create window with proper sizing
        window_name = 'GelSight Stream'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Calculate display dimensions
        display_width = int(self.width * self.display_scale)
        display_height = int(self.height * self.display_scale)
        cv2.resizeWindow(window_name, display_width, display_height)
        
        print(f"\nGelSight Data Streaming Started")
        print(f"Camera resolution: {self.width}x{self.height}")
        print(f"Display resolution: {display_width}x{display_height} (scale: {self.display_scale:.1f})")
        print("Controls:")
        print("  'q' or ESC: Quit")
        print("  'r': Start/stop recording")
        print("  's': Save current frame")
        print("  'c': Calibrate (save reference frame)")
        print("  SPACE: Pause/unpause")
        print("  '+'/'-': Increase/decrease display scale")
        print("\nStreaming live video...")
        
        try:
            while True:
                # Capture frame
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    print("Error: Could not read frame from camera")
                    break
                
                if frame.size == 0:
                    print("Error: Empty frame received")
                    continue
                
                # Process frame
                display_frame = self.process_frame(frame)
                
                # Record frame if recording
                if self.is_recording and not self.is_paused:
                    self.video_writer.write(frame)
                
                # Display frame (OpenCV will handle the scaling since we set WINDOW_NORMAL)
                cv2.imshow(window_name, display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                elif key == ord('r'):  # Toggle recording
                    if self.is_recording:
                        self.stop_recording()
                    else:
                        self.start_recording()
                elif key == ord('s'):  # Save frame
                    self.save_frame(frame)
                elif key == ord('c'):  # Calibrate
                    self.calibrate(frame)
                elif key == ord(' '):  # Pause/unpause
                    self.is_paused = not self.is_paused
                    print("Paused" if self.is_paused else "Resumed")
                elif key == ord('+') or key == ord('='):  # Increase scale
                    self.display_scale = min(2.0, self.display_scale + 0.1)
                    display_width = int(self.width * self.display_scale)
                    display_height = int(self.height * self.display_scale)
                    cv2.resizeWindow(window_name, display_width, display_height)
                    print(f"Display scale: {self.display_scale:.1f}")
                elif key == ord('-'):  # Decrease scale
                    self.display_scale = max(0.2, self.display_scale - 0.1)
                    display_width = int(self.width * self.display_scale)
                    display_height = int(self.height * self.display_scale)
                    cv2.resizeWindow(window_name, display_width, display_height)
                    print(f"Display scale: {self.display_scale:.1f}")
                
        except KeyboardInterrupt:
            print("\nStream interrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        if self.is_recording:
            self.stop_recording()
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        print("Resources cleaned up")

def main():
    # Create and start the streamer
    # Adjust parameters as needed:
    # - camera_index: change if your GelSight is on a different camera
    # - width/height: desired camera resolution (will use actual camera resolution)
    # - display_scale: scale factor for display window (0.5 = half size, 1.0 = full size)
    streamer = GelSightStreamer(
        camera_index=1, 
        width=640, 
        height=480, 
        display_scale=0.6  # Smaller default display for better fit
    )
    streamer.stream()

if __name__ == "__main__":
    main() 