import cv2
import numpy as np
import pytesseract

# Define Sudoku grid size
N = 9

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
                row_numbers.append(int(number[-1]))
            else:
                row_numbers.append(0)
        sudoku_grid.append(row_numbers)
    return sudoku_grid

def printing(arr):
    for i in range(N):
        for j in range(N):
            print(arr[i][j], end=" ")
        print()

def isSafe(grid, row, col, num):
    for x in range(9):
        if grid[row][x] == num or grid[x][col] == num:
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
    for num in range(1, N + 1):
        if isSafe(grid, row, col, num):
            grid[row][col] = num
            if solveSudoku(grid, row, col + 1):
                return True
            grid[row][col] = 0
    return False

def make_grid_solvable(grid):
    """
    Adjust the grid minimally to make it solvable.
    """
    temp_grid = [row[:] for row in grid]

    # Check if the grid is already solvable
    if solveSudoku(temp_grid, 0, 0):
        return grid  # No changes needed

    print("\nAdjusting grid to make it solvable...")

    # Identify and fix conflicting cells
    for row in range(N):
        for col in range(N):
            if grid[row][col] > 0:
                # Check if the number violates Sudoku rules
                num = grid[row][col]
                grid[row][col] = 0
                if not isSafe(grid, row, col, num):
                    grid[row][col] = 0  # Clear the conflicting cell
                else:
                    grid[row][col] = num  # Restore if it's safe

    # Try solving after clearing conflicts
    temp_grid = [row[:] for row in grid]
    if solveSudoku(temp_grid, 0, 0):
        return grid

    # Add minimal numbers to empty cells to ensure solvability
    for row in range(N):
        for col in range(N):
            if grid[row][col] == 0:
                for num in range(1, N + 1):
                    if isSafe(grid, row, col, num):
                        grid[row][col] = num
                        temp_grid = [row[:] for row in grid]
                        if solveSudoku(temp_grid, 0, 0):
                            return grid  # Return after minimal adjustment
                        grid[row][col] = 0  # Undo if not solvable

    return grid


def main(image_path):
    thresh = preprocess_image(image_path)
    cells = extract_cells(thresh)
    sudoku_grid = recognize_numbers(cells)

    print("Extracted Sudoku Grid:")
    printing(sudoku_grid)

    sudoku_grid = make_grid_solvable(sudoku_grid)

    if solveSudoku(sudoku_grid, 0, 0):
        print("\nSolved Sudoku Grid:")
        printing(sudoku_grid)
    else:
        print("Unable to solve, even after adjustments.")

if __name__ == "__main__":
    image_path = "cropped_grid.jpg"
    main(image_path)
