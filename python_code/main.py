import cv2
import numpy as np
import time

def detect_largest_quadrilateral(frame):
    """
    Detect the largest quadrilateral in the frame based on contour area.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = None
    max_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # Filter out small areas
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4 and area > max_area:  # Only consider quadrilaterals
                largest_contour = approx
                max_area = area

    if largest_contour is not None:
        return largest_contour
    return None

def draw_corners(frame, corners):
    """
    Draw red dots on the detected corners of the largest quadrilateral.
    """
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(frame, (x, y), 1, (255, 0, 255), -1)  # Red dot for each corner

def crop_to_quadrilateral(frame, corners):
    """
    Crop the image to the region inside the detected quadrilateral.
    """
    # Define the destination points for a straightened quadrilateral
    width = 500  # Set the desired width
    height = 500  # Set the desired height
    dst_points = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")

    # Sort the corners for perspective transform
    corners = corners.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = corners.sum(axis=1)
    rect[0] = corners[np.argmin(s)]
    rect[2] = corners[np.argmax(s)]
    diff = np.diff(corners, axis=1)
    rect[1] = corners[np.argmin(diff)]
    rect[3] = corners[np.argmax(diff)]

    # Compute the perspective transform matrix
    M = cv2.getPerspectiveTransform(rect, dst_points)
    cropped = cv2.warpPerspective(frame, M, (width, height))
    return cropped

def main():
    cap = cv2.VideoCapture(0)
    stable_start_time = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        corners = detect_largest_quadrilateral(frame)
        if corners is not None:
            draw_corners(frame, corners)

            # Check if corners are stable for 2 seconds
            if stable_start_time is None:
                stable_start_time = time.time()
            elif time.time() - stable_start_time >= 2:
                print("Stable detection for 2 seconds, capturing image!")
                cropped = crop_to_quadrilateral(frame, corners)
                cv2.imwrite("cropped_grid.jpg", cropped)
                cv2.imshow("Cropped Grid", cropped)
                cv2.waitKey(0)  # Wait for the user to close the new window
                break
        else:
            stable_start_time = None  # Reset if corners are no longer detected

        # Display the live feed
        cv2.imshow("Sudoku Grid Corner Detection", frame)

        key = cv2.waitKey(1)
        if key == 27:  # Escape key to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
