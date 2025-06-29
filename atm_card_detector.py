import cv2

# This is the Ideal aspect ratio of an ATM card (width:height)
CARD_ASPECT_RATIO = 1.586
ASPECT_RATIO_TOLERANCE = 0.3

# Adjusted area limits (pixels)
MIN_CARD_AREA = 3000
MAX_CARD_AREA = 100000

def is_card_shape(approx):
    if len(approx) == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        area = cv2.contourArea(approx)

        # Debug: Only print bigger shapes
        if area > 1000:
            print(f"Area: {int(area)}, Aspect Ratio: {round(aspect_ratio, 2)}")

        if (CARD_ASPECT_RATIO - ASPECT_RATIO_TOLERANCE < aspect_ratio < CARD_ASPECT_RATIO + ASPECT_RATIO_TOLERANCE) and (MIN_CARD_AREA < area < MAX_CARD_AREA):
            return True
    return False

# Starting the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Pre-processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 20, 100)  # More sensitive edge detection

    # Find contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected = False
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)

        # Draw all contours (blue)
        cv2.drawContours(frame, [approx], -1, (255, 0, 0), 1)

        # Check if it's card-like
        if is_card_shape(approx):
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)  # Highlight valid card
            cv2.putText(frame, "CARD DETECTED!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            detected = True
            break

    if not detected:
        cv2.putText(frame, "No card", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)

    # Show output
    cv2.imshow("ATM Card Detector", frame)

    # PRESS 'q' TO Exit 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
