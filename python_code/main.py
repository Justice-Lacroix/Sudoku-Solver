import cv2
import numpy as np
import time

# Function to detect the largest quadrilateral in a given frame
def detect_largest_quadrilateral(frame):
    # Convert the frame to grayscale (required for many image processing operations)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grayscale", gray)  # Show grayscale version

    # Apply a Gaussian Blur to reduce noise and detail in the image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imshow("Blurred", blurred)  # Show blurred version

    # Use the Canny edge detector to find edges in the blurred image
    edged = cv2.Canny(blurred, 50, 150)
    cv2.imshow("Edges", edged)  # Show edge-detected version

    # Find all contours in the edge-detected image
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = None  # To store the largest quadrilateral contour
    max_area = 0  # Track the largest area found

    # Iterate through all detected contours
    for contour in contours:
        # Calculate the contour area
        area = cv2.contourArea(contour)

        # Ignore small areas to reduce noise
        if area > 1000:
            # Calculate the arc length (perimeter) of the contour
            peri = cv2.arcLength(contour, True)

            # Approximate the contour to a polygon
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            # Check if the approximated polygon is a quadrilateral and larger than the current maximum area
            if len(approx) == 4 and area > max_area:
                largest_contour = approx
                max_area = area

    # Return the largest quadrilateral found, or None if no valid quadrilateral is detected
    if largest_contour is not None:
        return largest_contour
    return None

# Function to draw corners of a quadrilateral on the frame
def draw_corners(frame, corners):
    # Iterate over each corner point
    for corner in corners:
        x, y = corner.ravel()  # Flatten the corner array to get x, y coordinates
        # Draw a small circle at each corner (color: magenta)
        cv2.circle(frame, (x, y), 5, (255, 0, 255), -1)

# Function to crop the image to the detected quadrilateral
def crop_to_quadrilateral(frame, corners):
    # Desired output dimensions of the cropped quadrilateral
    width = 500
    height = 500

    # Define the destination points for a straightened quadrilateral
    dst_points = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")

    # Reshape the corners array for further manipulation
    corners = corners.reshape(4, 2)

    # Sort the corners for perspective transformation
    rect = np.zeros((4, 2), dtype="float32")
    s = corners.sum(axis=1)
    rect[0] = corners[np.argmin(s)]  # Top-left
    rect[2] = corners[np.argmax(s)]  # Bottom-right

    diff = np.diff(corners, axis=1)
    rect[1] = corners[np.argmin(diff)]  # Top-right
    rect[3] = corners[np.argmax(diff)]  # Bottom-left

    # Compute the perspective transform matrix
    M = cv2.getPerspectiveTransform(rect, dst_points)

    # Perform the perspective warp
    cropped = cv2.warpPerspective(frame, M, (width, height))
    return cropped

# Main function to capture live video, detect quadrilaterals, and crop
def main():
    # Open the webcam (device index 0)
    cap = cv2.VideoCapture(0)
    stable_start_time = None  # To track stability of the detected corners

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break  # Exit if the frame could not be read

        # Detect the largest quadrilateral in the frame
        corners = detect_largest_quadrilateral(frame)

        if corners is not None:
            # If a quadrilateral is detected, draw its corners
            draw_corners(frame, corners)

            # Check if the quadrilateral is stable for 2 seconds
            if stable_start_time is None:
                stable_start_time = time.time()
            elif time.time() - stable_start_time >= 2:
                # If stable for 2 seconds, capture and crop the quadrilateral
                print("Stable detection for 2 seconds, capturing image!")
                cropped = crop_to_quadrilateral(frame, corners)
                cv2.imwrite("cropped_grid.jpg", cropped)  # Save the cropped image
                cv2.imshow("Cropped Grid", cropped)  # Show the cropped image
                cv2.waitKey(0)  # Wait for user to close the window
                break
        else:
            # Reset stability timer if quadrilateral is not detected
            stable_start_time = None

        # Display the live video feed with corner overlays
        cv2.imshow("Sudoku Grid Corner Detection", frame)

        # Exit if the Escape key is pressed
        key = cv2.waitKey(1)
        if key == 27:
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Run the main function if the script is executed directly
if __name__ == "__main__":
    main()
