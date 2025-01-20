import cv2
import numpy as np
import pytesseract

def preprocess_image(image_path):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Apply adaptive thresholding to make the grid and numbers stand out
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Show the thresholded image
    cv2.imshow('Threshold Image', thresh)
    cv2.waitKey(0)  # Wait for any key to be pressed
    cv2.destroyAllWindows()

    return thresh

def extract_cells(thresh_img):
    # Find contours to locate the Sudoku grid
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest rectangle assumed to be the grid
    grid_contour = max(contours, key=cv2.contourArea)

    # Get a bounding box for the grid
    x, y, w, h = cv2.boundingRect(grid_contour)
    grid = thresh_img[y:y + h, x:x + w]

    # Resize the grid to a standard size (e.g., 450x450)
    grid_resized = cv2.resize(grid, (450, 450))

    # Show the resized grid
    cv2.imshow('Resized Grid', grid_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Divide the grid into 81 cells (9x9)
    cells = []
    cell_size = 50  # 450/9 = 50 pixels per cell
    for row in range(9):
        row_cells = []
        for col in range(9):
            x_start = col * cell_size
            y_start = row * cell_size
            cell = grid_resized[y_start:y_start + cell_size, x_start:x_start + cell_size]
            row_cells.append(cell)

            # Show each extracted cell
            cv2.imshow(f'Cell {row+1}-{col+1}', cell)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        cells.append(row_cells)
    
    return cells

def recognize_numbers(cells):
    sudoku_grid = []
    for row in cells:
        row_numbers = []
        for cell in row:
            # Invert the cell for better OCR
            inverted_cell = cv2.bitwise_not(cell)

            # Use Tesseract OCR to recognize the digit
            number = pytesseract.image_to_string(inverted_cell, config='--psm 10 digits')

            # Clean up OCR output
            number = number.strip()

            if number.isdigit():
                # If number > 9, extract the last digit
                row_numbers.append(int(number[-1]))
            else:
                row_numbers.append(0)  # Use 0 for empty cells
        sudoku_grid.append(row_numbers)
    return sudoku_grid

def main(image_path):
    # Preprocess the image
    thresh = preprocess_image(image_path)

    # Extract individual cells
    cells = extract_cells(thresh)

    # Recognize numbers and construct the Sudoku grid
    sudoku_grid = recognize_numbers(cells)

    # Print the grid
    for row in sudoku_grid:
        print(row)

if __name__ == "__main__":
    # Replace 'cropped_grid.jpg' with your image path
    image_path = "cropped_grid.jpg"
    main(image_path)
