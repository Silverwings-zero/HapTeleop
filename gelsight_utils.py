import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path

"""
GelSight Utilities

This module provides utility functions for processing and analyzing GelSight tactile sensor data.
Functions include:
- Image preprocessing
- Contact detection
- Force estimation
- Data visualization
- Batch processing of recorded data
"""

class GelSightProcessor:
    def __init__(self, reference_frame=None):
        self.reference_frame = reference_frame
        
    def load_reference(self, reference_path):
        """Load a reference frame from file."""
        if os.path.exists(reference_path):
            self.reference_frame = cv2.imread(reference_path)
            print(f"Reference frame loaded from {reference_path}")
            return True
        else:
            print(f"Error: Reference frame not found at {reference_path}")
            return False
    
    def preprocess_frame(self, frame, blur_kernel=5, enhance_contrast=True):
        """Preprocess a frame for analysis."""
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        # Apply Gaussian blur to reduce noise
        if blur_kernel > 0:
            gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
        
        # Enhance contrast if requested
        if enhance_contrast:
            gray = cv2.equalizeHist(gray)
        
        return gray
    
    def compute_difference(self, current_frame, reference_frame=None):
        """Compute difference between current frame and reference."""
        if reference_frame is None:
            reference_frame = self.reference_frame
        
        if reference_frame is None:
            print("Warning: No reference frame available")
            return None
        
        # Preprocess both frames
        current_gray = self.preprocess_frame(current_frame)
        reference_gray = self.preprocess_frame(reference_frame)
        
        # Compute absolute difference
        diff = cv2.absdiff(current_gray, reference_gray)
        
        return diff
    
    def detect_contact(self, frame, threshold=30, min_area=100):
        """Detect contact areas in the frame."""
        if self.reference_frame is None:
            print("Warning: No reference frame for contact detection")
            return None, []
        
        # Compute difference
        diff = self.compute_difference(frame)
        if diff is None:
            return None, []
        
        # Apply threshold
        _, binary = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        
        return binary, valid_contours
    
    def estimate_force_magnitude(self, frame, contact_mask=None):
        """Estimate relative force magnitude from image intensity changes."""
        if self.reference_frame is None:
            print("Warning: No reference frame for force estimation")
            return 0
        
        diff = self.compute_difference(frame)
        if diff is None:
            return 0
        
        if contact_mask is not None:
            # Only consider areas within the contact mask
            masked_diff = cv2.bitwise_and(diff, contact_mask)
            force_magnitude = np.sum(masked_diff) / np.count_nonzero(contact_mask) if np.count_nonzero(contact_mask) > 0 else 0
        else:
            force_magnitude = np.mean(diff)
        
        return force_magnitude
    
    def analyze_frame(self, frame, visualize=False):
        """Comprehensive analysis of a single frame."""
        results = {
            'contact_detected': False,
            'num_contacts': 0,
            'total_contact_area': 0,
            'force_magnitude': 0,
            'contact_centers': []
        }
        
        # Detect contacts
        binary_mask, contours = self.detect_contact(frame)
        
        if binary_mask is not None:
            results['contact_detected'] = len(contours) > 0
            results['num_contacts'] = len(contours)
            
            # Calculate contact areas and centers
            for contour in contours:
                area = cv2.contourArea(contour)
                results['total_contact_area'] += area
                
                # Calculate center of mass
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    results['contact_centers'].append((cx, cy))
            
            # Estimate force
            results['force_magnitude'] = self.estimate_force_magnitude(frame, binary_mask)
        
        if visualize:
            self.visualize_analysis(frame, binary_mask, contours, results)
        
        return results
    
    def visualize_analysis(self, frame, contact_mask, contours, results):
        """Visualize the analysis results."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original frame
        axes[0, 0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Frame')
        axes[0, 0].axis('off')
        
        # Difference image
        if self.reference_frame is not None:
            diff = self.compute_difference(frame)
            if diff is not None:
                axes[0, 1].imshow(diff, cmap='gray')
                axes[0, 1].set_title('Difference from Reference')
                axes[0, 1].axis('off')
        
        # Contact mask
        if contact_mask is not None:
            axes[1, 0].imshow(contact_mask, cmap='gray')
            axes[1, 0].set_title('Contact Mask')
            axes[1, 0].axis('off')
        
        # Annotated frame with contacts
        annotated = frame.copy()
        if contours:
            cv2.drawContours(annotated, contours, -1, (0, 255, 0), 2)
            for center in results['contact_centers']:
                cv2.circle(annotated, center, 5, (255, 0, 0), -1)
        
        axes[1, 1].imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title(f"Contacts Detected: {results['num_contacts']}")
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()

def batch_process_video(video_path, reference_frame_path=None, output_dir=None):
    """Process a recorded video file and extract tactile data."""
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return None
    
    # Initialize processor
    processor = GelSightProcessor()
    if reference_frame_path:
        processor.load_reference(reference_frame_path)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return None
    
    # Setup output directory
    if output_dir is None:
        output_dir = f"analysis_{Path(video_path).stem}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process frames
    frame_results = []
    frame_count = 0
    
    print("Processing video frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Analyze frame
        results = processor.analyze_frame(frame)
        results['frame_number'] = frame_count
        results['timestamp'] = frame_count / cap.get(cv2.CAP_PROP_FPS)
        frame_results.append(results)
        
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...")
    
    cap.release()
    
    # Save results
    results_file = os.path.join(output_dir, "analysis_results.txt")
    with open(results_file, 'w') as f:
        f.write("Frame,Timestamp,ContactDetected,NumContacts,TotalArea,ForceMagnitude\n")
        for result in frame_results:
            f.write(f"{result['frame_number']},{result['timestamp']:.3f},"
                   f"{result['contact_detected']},{result['num_contacts']},"
                   f"{result['total_contact_area']:.2f},{result['force_magnitude']:.2f}\n")
    
    print(f"Analysis complete. Results saved to {results_file}")
    return frame_results

def plot_force_timeline(results, save_path=None):
    """Plot force magnitude over time."""
    timestamps = [r['timestamp'] for r in results]
    forces = [r['force_magnitude'] for r in results]
    contacts = [r['contact_detected'] for r in results]
    
    plt.figure(figsize=(12, 6))
    
    # Plot force magnitude
    plt.subplot(2, 1, 1)
    plt.plot(timestamps, forces, 'b-', linewidth=1)
    plt.ylabel('Force Magnitude')
    plt.title('GelSight Force Analysis')
    plt.grid(True, alpha=0.3)
    
    # Plot contact detection
    plt.subplot(2, 1, 2)
    plt.plot(timestamps, contacts, 'r-', linewidth=2)
    plt.ylabel('Contact Detected')
    plt.xlabel('Time (seconds)')
    plt.ylim(-0.1, 1.1)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def main():
    """Example usage of GelSight utilities."""
    print("GelSight Utilities")
    print("Available functions:")
    print("- batch_process_video(video_path, reference_frame_path)")
    print("- plot_force_timeline(results)")
    print("- GelSightProcessor class for real-time analysis")
    
    # Example: Process a video file if it exists
    example_video = "gelsight_data/gelsight_recording_latest.avi"
    example_reference = "gelsight_data/gelsight_reference_latest.png"
    
    if os.path.exists(example_video):
        print(f"\nProcessing example video: {example_video}")
        results = batch_process_video(example_video, example_reference)
        if results:
            plot_force_timeline(results)

if __name__ == "__main__":
    main() 