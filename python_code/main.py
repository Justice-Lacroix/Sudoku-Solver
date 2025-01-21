import cv2
import numpy as np
import time
import Getnumbers_Solve  # Sudoku-oplossing importeren


# Functie om de grootste vierhoek (het Sudoku-raster) in een afbeelding te detecteren
def detect_largest_quadrilateral(frame):
    """
    Detecteert de grootste vierhoek in een afbeelding en retourneert de hoeken.
    Dit wordt gedaan door:
    1. De afbeelding om te zetten naar grijswaarden.
    2. Ruis te verminderen met een Gaussiaans filter.
    3. Randen te detecteren met Canny edge detection.
    4. Contouren te vinden en de grootste vierhoek te selecteren.
    """

    # 1. Omzetten naar grijswaarden
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. Ruis verminderen met Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Randen detecteren met Canny
    edged = cv2.Canny(blurred, 50, 150)

    # Toon de tussenstappen voor debugging
    cv2.imshow("Grayscale", gray)
    cv2.imshow("Blurred", blurred)
    cv2.imshow("Edges", edged)

    # 4. Contouren zoeken in de afbeelding
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = None
    max_area = 0

    # 5. Doorloop alle gevonden contouren
    for contour in contours:
        area = cv2.contourArea(contour)

        # Alleen grote contouren overwegen
        if area > 1000:
            peri = cv2.arcLength(contour, True)  # Bereken de omtrek
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)  # Benader contour

            # Selecteer de grootste vierhoek
            if len(approx) == 4 and area > max_area:
                largest_contour = approx
                max_area = area

    # Retourneer de grootste vierhoek of None als er geen gevonden is
    return largest_contour if largest_contour is not None else None


# Functie om hoeken op de afbeelding te markeren
def draw_corners(frame, corners):
    """
    Teken kleine cirkels op de vier hoeken van het gedetecteerde Sudoku-raster.
    """
    for corner in corners:
        x, y = corner.ravel()  # Haal de x- en y-coördinaten op
        cv2.circle(frame, (x, y), 5, (255, 0, 255), -1)  # Magenta cirkel tekenen


# Functie om het Sudoku-raster bij te snijden en recht te zetten
def crop_to_quadrilateral(frame, corners):
    """
    Snijdt het Sudoku-raster uit de afbeelding en corrigeert het perspectief.
    """
    width, height = 500, 500  # Standaard grootte voor de uiteindelijke afbeelding

    # Doelpuntcoördinaten voor het rechtgetrokken beeld
    dst_points = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")

    # Zorg ervoor dat de hoekpunten correct zijn gerangschikt
    corners = corners.reshape(4, 2)

    # 1. Sorteer de hoeken in de volgorde: linksboven, rechtsboven, rechtsonder, linksonder
    rect = np.zeros((4, 2), dtype="float32")
    s = corners.sum(axis=1)
    rect[0] = corners[np.argmin(s)]  # Links boven
    rect[2] = corners[np.argmax(s)]  # Rechts onder

    diff = np.diff(corners, axis=1)
    rect[1] = corners[np.argmin(diff)]  # Rechts boven
    rect[3] = corners[np.argmax(diff)]  # Links onder

    # 2. Bereken de perspectieftransformatie
    M = cv2.getPerspectiveTransform(rect, dst_points)

    # 3. Pas de transformatie toe
    cropped = cv2.warpPerspective(frame, M, (width, height))

    return cropped


# Functie om de camera te gebruiken en een Sudoku-raster te detecteren
def use_camera():
    """
    Opent de webcam, zoekt naar een Sudoku-raster, en slaat de uitgesneden afbeelding op.
    Wacht tot het raster 4 seconden stabiel is voordat de afbeelding wordt vastgelegd.
    """
    cap = cv2.VideoCapture(0)  # Open de webcam
    stable_start_time = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Stop als er geen frame wordt gelezen

        corners = detect_largest_quadrilateral(frame)

        if corners is not None:
            draw_corners(frame, corners)

            # Wacht tot het raster stabiel is voor 4 seconden
            if stable_start_time is None:
                stable_start_time = time.time()
            elif time.time() - stable_start_time >= 4:
                print("Stabiele detectie gedurende 4 seconden, afbeelding wordt vastgelegd!")
                cropped = crop_to_quadrilateral(frame, corners)
                cv2.imwrite("cropped_grid.jpg", cropped)  # Opslaan als afbeelding
                cv2.imshow("Cropped Grid", cropped)
                cv2.waitKey(0)
                cap.release()
                cv2.destroyAllWindows()
                return "cropped_grid.jpg"

        else:
            stable_start_time = None  # Reset timer als het raster verdwijnt

        cv2.imshow("Sudoku Raster Detectie", frame)

        # Stop als de Escape-toets wordt ingedrukt
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    return None


# Functie om een opgeslagen afbeelding te verwerken
def process_saved_image():
    """
    Vraagt de gebruiker om een bestandsnaam en verwerkt de afbeelding met Getnumbers_Solve.
    """
    filename = input("Voer de bestandsnaam van de afbeelding in (bijv. sudoku.jpg): ")
    Getnumbers_Solve.main(filename)  # Verwerk de afbeelding met Getnumbers_Solve


# Hoofdfunctie met keuzeoptie
def main():
    """
    Biedt de gebruiker de keuze tussen:
    1. Sudoku detecteren met de camera
    2. Sudoku detecteren vanuit een opgeslagen afbeelding
    """
    print("Kies een optie:")
    print("1 - Gebruik de camera")
    print("2 - Gebruik een opgeslagen afbeelding")

    choice = input("Voer 1 of 2 in: ").strip()

    if choice == "1":
        image_path = use_camera()
        if image_path is not None:
            Getnumbers_Solve.main(image_path)  # Verwerk de afbeelding
    elif choice == "2":
        process_saved_image()
    else:
        print("Ongeldige invoer, programma wordt afgesloten.")
        return


# Start het programma
if __name__ == "__main__":
    main()
