import cv2
import numpy as np
import time
import Getnumbers_Solve  # Sudoku-oplossing importeren


# Functie om de grootste vierhoek te detecteren
def detect_largest_quadrilateral(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    cv2.imshow("Grayscale", gray)
    cv2.imshow("Blurred", blurred)
    cv2.imshow("Edges", edged)

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

    return largest_contour if largest_contour is not None else None


# Functie om hoeken te tekenen
def draw_corners(frame, corners):
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(frame, (x, y), 5, (255, 0, 255), -1)  # Magenta cirkels


# Functie om een afbeelding te croppen en perspectiefcorrectie toe te passen
def crop_to_quadrilateral(frame, corners):
    width, height = 500, 500
    dst_points = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")

    corners = corners.reshape(4, 2)

    rect = np.zeros((4, 2), dtype="float32")
    s = corners.sum(axis=1)
    rect[0] = corners[np.argmin(s)]  # Links boven
    rect[2] = corners[np.argmax(s)]  # Rechts onder

    diff = np.diff(corners, axis=1)
    rect[1] = corners[np.argmin(diff)]  # Rechts boven
    rect[3] = corners[np.argmax(diff)]  # Links onder

    M = cv2.getPerspectiveTransform(rect, dst_points)
    cropped = cv2.warpPerspective(frame, M, (width, height))
    return cropped


# Functie voor cameragebruik (exact zoals in jouw code)
def use_camera():
    cap = cv2.VideoCapture(0)  # Open de webcam
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
            elif time.time() - stable_start_time >= 4:
                print("Stabiele detectie gedurende 4 seconden, afbeelding wordt vastgelegd!")
                cropped = crop_to_quadrilateral(frame, corners)
                cv2.imwrite("cropped_grid.jpg", cropped)
                cv2.imshow("Cropped Grid", cropped)
                cv2.waitKey(0)
                cap.release()
                cv2.destroyAllWindows()
                return "cropped_grid.jpg"

        else:
            stable_start_time = None

        cv2.imshow("Sudoku Raster Detectie", frame)

        if cv2.waitKey(1) == 27:  # Escape-toets om af te sluiten
            break

    cap.release()
    cv2.destroyAllWindows()
    return None


# Functie om een opgeslagen afbeelding te verwerken
def process_saved_image():
    filename = input("Voer de bestandsnaam van de afbeelding in (bijv. sudoku.jpg): ")
    Getnumbers_Solve.main(filename)  # Verwerk de afbeelding met Getnumbers_Solve


# Hoofdfunctie met keuzeoptie
def main():
    print("Kies een optie:")
    print("1 - Gebruik de camera")
    print("2 - Gebruik een opgeslagen afbeelding")

    choice = input("Voer 1 of 2 in: ").strip()

    if choice == "1":
        image_path = use_camera()
        if image_path is not None:
            Getnumbers_Solve.main(image_path)
    elif choice == "2":
        process_saved_image()
    else:
        print("Ongeldige invoer, programma wordt afgesloten.")
        return


# Start het programma
if __name__ == "__main__":
    main()
