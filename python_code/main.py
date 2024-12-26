import cv2
import numpy as np
import pytesseract

# Sudoku solver functions
N = 9

def printing(arr):
    for i in range(N):
        for j in range(N):
            print(arr[i][j], end=" ")
        print()

def isSafe(grid, row, col, num):
    for x in range(9):
        if grid[row][x] == num:
            return False
    for x in range(9):
        if grid[x][col] == num:
            return False
    startRow = row - row % 3
    startCol = col - col % 3
    for i in range(3):
        for j in range(3):
            if grid[i + startRow][j + startCol] == num:
                return False
    return True

def solveSudoku(grid, row, col):
    if row == N - 1 and col == N:
        return True
    if col == N:
        row += 1
        col = 0
    if grid[row][col] > 0:
        return solveSudoku(grid, row, col + 1)
    for num in range(1, N + 1, 1):
        if isSafe(grid, row, col, num):
            grid[row][col] = num
            if solveSudoku(grid, row, col + 1):
                return True
        grid[row][col] = 0
    return False

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    return thresh

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

def recognize_numbers(cells):
    sudoku_grid = []
    for row in cells:
        row_numbers = []
        for cell in row:
            inverted_cell = cv2.bitwise_not(cell)
            number = pytesseract.image_to_string(inverted_cell, config='--psm 10 digits')
            number = number.strip()
            if number.isdigit():
                row_numbers.append(int(number))
            else:
                row_numbers.append(0)
        sudoku_grid.append(row_numbers)
    return sudoku_grid

def solve_and_display(image_path):
    thresh = preprocess_image(image_path)
    cells = extract_cells(thresh)
    sudoku_grid = recognize_numbers(cells)
    print("Extracted Sudoku Grid:")
    printing(sudoku_grid)
    if solveSudoku(sudoku_grid, 0, 0):
        print("\nSolved Sudoku Grid:")
        printing(sudoku_grid)
    else:
        print("\nNo solution exists.")

# Detection and Cropping Functions
def detect_largest_quadrilateral(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = None
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4 and area > max_area:
                largest_contour = approx
                max_area = area
    if largest_contour is not None:
        return largest_contour
    return None

def draw_corners(frame, corners):
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

def crop_to_quadrilateral(frame, corners):
    width, height = 500, 500
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
                stable_start_time = cv2.getTickCount()
            else:
                elapsed_time = (cv2.getTickCount() - stable_start_time) / cv2.getTickFrequency()
                if elapsed_time >= 2:
                    print("Stable detection for 2 seconds, capturing image!")
                    cropped = crop_to_quadrilateral(frame, corners)
                    cv2.imwrite("cropped_grid.jpg", cropped)
                    solve_and_display("cropped_grid.jpg")
                    break
        else:
            stable_start_time = None

        cv2.imshow("Sudoku Grid Corner Detection", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
