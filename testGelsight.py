import cv2

""" 
This code will write 'hello.jpg' to the directory the script is run in.
This code runs on Windows.
This code requires the opencv-python package to be installed.
"""

def saveframe(filepath):
    # Open Video device using OpenCV2 and Microsoft Media Foundation.
    cap = cv2.VideoCapture(1, cv2.CAP_MSMF)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera with index 1. Trying index 0...")
        cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
        if not cap.isOpened():
            print("Error: Could not open any camera. Please check if a camera is connected.")
            return False
    
    # Give camera time to initialize
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Capture a frame
    ret, frame = cap.read()
    
    # Check if frame was captured successfully
    if not ret or frame is None:
        print("Error: Could not capture frame from camera.")
        cap.release()
        return False
    
    # Check if frame is not empty
    if frame.size == 0:
        print("Error: Captured frame is empty.")
        cap.release()
        return False
    
    # Save the frame to a file
    try:
        status = cv2.imwrite(filepath, frame)
        if status:
            print(f"Successfully saved frame to {filepath}")
        else:
            print(f"Failed to save frame to {filepath}")
    except Exception as e:
        print(f"Error saving frame: {e}")
        status = False
    
    # Release the camera
    cap.release()
    return status

if __name__ == '__main__':
    saveframe('Hello.jpg')
