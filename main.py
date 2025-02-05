import cv2
import numpy as np
import time

def create_background(cap, num_frames=100):
    """
    Captures a smooth background by averaging multiple frames.
    """
    print("Capturing background. Please move out of frame.")
    avg_background = None

    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame {i+1}/{num_frames}")
            continue
        
        frame = cv2.GaussianBlur(frame, (7, 7), 0)  # Smoother background
        if avg_background is None:
            avg_background = np.float32(frame)
        else:
            cv2.accumulateWeighted(frame, avg_background, 0.02)  # Running average
        
        time.sleep(0.02)

    return cv2.convertScaleAbs(avg_background)  # Convert to uint8


def create_mask(frame, lower_color, upper_color):
    """
    Creates a refined mask for the given color range.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Use a bilateral filter to smooth the colors and remove noise
    hsv = cv2.bilateralFilter(hsv, 9, 75, 75)

    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Morphological operations to remove small noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=2)

    return mask


def apply_cloak_effect(frame, mask, background):
    """
    Combines the frame and background based on the mask.
    """
    mask_inv = cv2.bitwise_not(mask)

    fg = cv2.bitwise_and(frame, frame, mask=mask_inv)  # Keeps everything except the cloak
    bg = cv2.bitwise_and(background, background, mask=mask)  # Fills the cloak with background

    blended = cv2.addWeighted(fg, 1, bg, 1, 0)  # Blending to smooth the transition
    return cv2.GaussianBlur(blended, (5, 5), 0)  # Final smoothing


def main():
    print("OpenCV version:", cv2.__version__)

    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    try:
        background = create_background(cap)
    except ValueError as e:
        print(f"Error: {e}")
        cap.release()
        return

    lower_blue = np.array([90, 50, 50])  # Adjust these values for better detection
    upper_blue = np.array([130, 255, 255])

    print("Starting main loop. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            time.sleep(1)
            continue

        frame = cv2.flip(frame, 1)  # Flip for mirror effect
        mask = create_mask(frame, lower_blue, upper_blue)
        result = apply_cloak_effect(frame, mask, background)

        cv2.imshow('Invisible Cloak Effect', result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
