import cv2
import pytesseract

# Define Sudoku grid size
N = 9  # The Sudoku grid is a 9x9 matrix


# Function to preprocess the input image for better number recognition
def preprocess_image(image_path):
    # Read the image in grayscale mode
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    # Apply adaptive thresholding for better contrast (black and white)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    return thresh


# Function to extract individual cells from the Sudoku grid
def extract_cells(thresh_img):
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Identify the largest contour as the grid
    grid_contour = max(contours, key=cv2.contourArea)
    # Get the bounding box around the largest contour
    x, y, w, h = cv2.boundingRect(grid_contour)
    # Crop the grid from the thresholded image
    grid = thresh_img[y:y + h, x:x + w]
    # Resize the cropped grid to a standard size (450x450 pixels)
    grid_resized = cv2.resize(grid, (450, 450))

    # Divide the grid into 9x9 cells
    cells = []
    cell_size = 50  # Each cell is 50x50 pixels in the resized grid
    for row in range(9):
        row_cells = []
        for col in range(9):
            # Extract individual cells using slicing
            x_start = col * cell_size
            y_start = row * cell_size
            cell = grid_resized[y_start:y_start + cell_size, x_start:x_start + cell_size]
            row_cells.append(cell)
        cells.append(row_cells)
    return cells


# Function to recognize numbers from the extracted cells
def recognize_numbers(cells):
    sudoku_grid = []  # To store the extracted Sudoku numbers
    for row in cells:
        row_numbers = []
        for cell in row:
            # Invert the cell image for better OCR results
            inverted_cell = cv2.bitwise_not(cell)
            # Use Tesseract OCR to recognize the number in the cell
            number = pytesseract.image_to_string(inverted_cell, config='--psm 10 digits')
            number = number.strip()  # Remove any extra spaces or newline characters
            if number.isdigit():
                # Append the recognized digit
                row_numbers.append(int(number[-1]))
            else:
                # If no number is recognized, treat it as 0 (empty cell)
                row_numbers.append(0)
        sudoku_grid.append(row_numbers)
    return sudoku_grid


# Function to print a 2D array (used to display the Sudoku grid)
def printing(arr):
    for i in range(N):
        for j in range(N):
            print(arr[i][j], end=" ")
        print()


# Function to check if a number can be safely placed in a cell
def isSafe(grid, row, col, num):
    # Check the row and column for duplicates
    for x in range(9):
        if grid[row][x] == num or grid[x][col] == num:
            return False
    # Check the 3x3 subgrid for duplicates
    startRow = row - row % 3
    startCol = col - col % 3
    for i in range(3):
        for j in range(3):
            if grid[i + startRow][j + startCol] == num:
                return False
    return True


# Recursive function to solve the Sudoku grid
def solveSudoku(grid, row, col):
    # If we reach the end of the grid, the Sudoku is solved
    if row == N - 1 and col == N:
        return True
    # Move to the next row if we are at the end of a column
    if col == N:
        row += 1
        col = 0
    # Skip pre-filled cells
    if grid[row][col] > 0:
        return solveSudoku(grid, row, col + 1)
    # Try placing numbers 1-9 in the cell
    for num in range(1, N + 1):
        if isSafe(grid, row, col, num):
            grid[row][col] = num
            if solveSudoku(grid, row, col + 1):
                return True
            grid[row][col] = 0  # Backtrack if placing the number doesn't lead to a solution
    return False


# Function to adjust the grid to make it solvable, if necessary
def make_grid_solvable(grid):
    """
    Adjust the grid minimally to make it solvable.
    """
    temp_grid = [row[:] for row in grid]  # Create a copy of the grid

    # Check if the grid is already solvable
    if solveSudoku(temp_grid, 0, 0):
        return grid  # No changes needed

    print("\nAdjusting grid to make it solvable...")

    # Identify and fix conflicting cells
    for row in range(N):
        for col in range(N):
            if grid[row][col] > 0:
                num = grid[row][col]
                grid[row][col] = 0
                if not isSafe(grid, row, col, num):
                    grid[row][col] = 0  # Clear conflicting cell
                else:
                    grid[row][col] = num

    # Try solving after clearing conflicts
    temp_grid = [row[:] for row in grid]
    if solveSudoku(temp_grid, 0, 0):
        return grid

    # Add minimal numbers to ensure solvability
    for row in range(N):
        for col in range(N):
            if grid[row][col] == 0:
                for num in range(1, N + 1):
                    if isSafe(grid, row, col, num):
                        grid[row][col] = num
                        temp_grid = [row[:] for row in grid]
                        if solveSudoku(temp_grid, 0, 0):
                            return grid
                        grid[row][col] = 0

    return grid


# Main function to process the Sudoku image, extract the grid, and solve it
def main(image_path):
    # Preprocess the image to extract the grid
    thresh = preprocess_image(image_path)
    # Extract individual cells from the grid
    cells = extract_cells(thresh)
    # Recognize numbers from the cells
    sudoku_grid = recognize_numbers(cells)

    print("Extracted Sudoku Grid:")
    printing(sudoku_grid)

    # Adjust the grid and solve it
    sudoku_grid = make_grid_solvable(sudoku_grid)

    if solveSudoku(sudoku_grid, 0, 0):
        print("\nSolved Sudoku Grid:")
        printing(sudoku_grid)
    else:
        print("Unable to solve, even after adjustments.")


# Execute the main function when the script is run directly
if __name__ == "__main__":
    image_path = "cropped_grid.jpg"  # Input image path
    main(image_path)
