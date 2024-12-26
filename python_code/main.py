import cv2
import numpy as np
import pytesseract
import time

# Function to detect the largest quadrilateral in the frame
def detect_largest_quadrilateral(frame):
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

# Function to draw red dots on the detected corners
def draw_corners(frame, corners):
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(frame, (x, y), 1, (255, 0, 255), -1)  # Red dot for each corner

# Function to crop the image to the detected quadrilateral
def crop_to_quadrilateral(frame, corners):
    width = 500
    height = 500
    dst_points = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")

    corners = corners.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = corners.sum(axis=1)
    rect[0] = corners[np.argmin(s)]
    rect[2] = corners[np.argmax(s)]
    diff = np.diff(corners, axis=1)
    rect[1] = corners[np.argmin(diff)]
    rect[3] = corners[np.argmax(diff)]

    M = cv2.getPerspectiveTransform(rect, dst_points)
    cropped = cv2.warpPerspective(frame, M, (width, height))
    return cropped

# Function to preprocess the image for OCR
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    return thresh

# Function to extract individual cells from the Sudoku grid
def extract_cells(thresh_img):
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    grid_contour = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(grid_contour)
    grid = thresh_img[y:y + h, x:x + w]
    grid_resized = cv2.resize(grid, (450, 450))

    cells = []
    cell_size = 50
    for row in range(9):
        row_cells = []
        for col in range(9):
            x_start = col * cell_size
            y_start = row * cell_size
            cell = grid_resized[y_start:y_start + cell_size, x_start:x_start + cell_size]
            row_cells.append(cell)
        cells.append(row_cells)
    return cells

# Function to recognize numbers from cells using OCR
def recognize_numbers(cells):
    sudoku_grid = []
    for row in cells:
        row_numbers = []
        for cell in row:
            inverted_cell = cv2.bitwise_not(cell)
            number = pytesseract.image_to_string(inverted_cell, config='--psm 10 digits')
            number = number.strip()

            if number.isdigit():
                row_numbers.append(int(number[-1]))
            else:
                row_numbers.append(0)
        sudoku_grid.append(row_numbers)
    return sudoku_grid

# Main function to capture video and process the Sudoku grid
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

            if stable_start_time is None:
                stable_start_time = time.time()
            elif time.time() - stable_start_time >= 2:
                print("Stable detection for 2 seconds, capturing image!")
                cropped = crop_to_quadrilateral(frame, corners)
                cropped_filename = "cropped_grid.jpg"
                cv2.imwrite(cropped_filename, cropped)
                cv2.imshow("Cropped Grid", cropped)
                cv2.waitKey(0)

                # Process the Sudoku grid
                print("Processing the Sudoku grid...")
                try:
                    thresh = preprocess_image(cropped_filename)
                    cells = extract_cells(thresh)
                    sudoku_grid = recognize_numbers(cells)

                    # Print the extracted Sudoku grid
                    print("Extracted Sudoku Grid:")
                    for row in sudoku_grid:
                        print(row)
                except Exception as e:
                    print(f"An error occurred during Sudoku grid extraction: {e}")

                break
        else:
            stable_start_time = None

        cv2.imshow("Sudoku Grid Corner Detection", frame)

        key = cv2.waitKey(1)
        if key == 27:  # Escape key to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
