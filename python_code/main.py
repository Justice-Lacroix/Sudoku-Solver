import cv2
import numpy as np
import time

import Getnumbers_Solve  # Import module for further processing


# Function to detect the largest quadrilateral in a given frame
def detect_largest_quadrilateral(frame):
    # Convert the frame to grayscale (needed for edge detection and thresholding)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grayscale", gray)  # Show grayscale version

    # Apply Gaussian Blur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imshow("Blurred", blurred)  # Show blurred version

    # Detect edges using Canny edge detection
    edged = cv2.Canny(blurred, 50, 150)
    cv2.imshow("Edges", edged)  # Show edge-detected version

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = None  # Store the largest quadrilateral contour
    max_area = 0  # Variable to track the largest area found

    # Iterate through detected contours
    for contour in contours:
        area = cv2.contourArea(contour)  # Calculate contour area

        # Ignore small areas to reduce noise
        if area > 1000:
            peri = cv2.arcLength(contour, True)  # Calculate perimeter
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)  # Approximate contour shape

            # Check if the contour is a quadrilateral and larger than the previous one
            if len(approx) == 4 and area > max_area:
                largest_contour = approx
                max_area = area

    # Return the largest detected quadrilateral, or None if none found
    return largest_contour if largest_contour is not None else None


# Function to draw small circles on detected corners
def draw_corners(frame, corners):
    for corner in corners:
        x, y = corner.ravel()  # Get x, y coordinates
        cv2.circle(frame, (x, y), 5, (255, 0, 255), -1)  # Draw magenta circles at each corner


# Function to crop the detected quadrilateral and warp it to a fixed size
def crop_to_quadrilateral(frame, corners):
    width, height = 500, 500  # Define output image size

    dst_points = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")

    corners = corners.reshape(4, 2)  # Ensure proper shape of corner points

    # Sort the corner points in order: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    s = corners.sum(axis=1)
    rect[0] = corners[np.argmin(s)]  # Top-left
    rect[2] = corners[np.argmax(s)]  # Bottom-right

    diff = np.diff(corners, axis=1)
    rect[1] = corners[np.argmin(diff)]  # Top-right
    rect[3] = corners[np.argmax(diff)]  # Bottom-left

    # Compute perspective transformation matrix
    M = cv2.getPerspectiveTransform(rect, dst_points)

    # Apply perspective warp to obtain a straightened version of the quadrilateral
    cropped = cv2.warpPerspective(frame, M, (width, height))
    return cropped


# Main function to capture live video, detect quadrilaterals, and crop the image
def main():
    cap = cv2.VideoCapture(0)  # Open webcam (device index 0)
    stable_start_time = None  # Track how long a quadrilateral remains stable

    while True:
        ret, frame = cap.read()  # Capture a frame from the webcam
        if not ret:
            break  # Exit if frame cannot be read

        corners = detect_largest_quadrilateral(frame)  # Detect quadrilateral

        if corners is not None:
            draw_corners(frame, corners)  # Draw detected corners

            # Check if the quadrilateral has remained stable for 4 seconds
            if stable_start_time is None:
                stable_start_time = time.time()
            elif time.time() - stable_start_time >= 4:
                print("Stable detection for 4 seconds, capturing image!")
                cropped = crop_to_quadrilateral(frame, corners)  # Crop detected quadrilateral
                cv2.imwrite("cropped_grid.jpg", cropped)  # Save cropped image
                cv2.imshow("Cropped Grid", cropped)  # Display cropped image
                cv2.waitKey(0)  # Wait for user input before proceeding
                break  # Exit loop after capturing image
        else:
            stable_start_time = None  # Reset stability timer if detection fails

        cv2.imshow("Sudoku Grid Corner Detection", frame)  # Display live feed

        # Exit when the Escape key (27) is pressed
        if cv2.waitKey(1) == 27:
            break

    cap.release()  # Release webcam
    cv2.destroyAllWindows()  # Close all OpenCV windows


# Run the main function if the script is executed directly
if __name__ == "__main__":
    main()
    Getnumbers_Solve.main("cropped_grid.jpg")  # Process the captured image
