#git clone https://github.com/Majdawad88/extracts-blue-color-objects-from-image.git
from picamera2 import Picamera2
import cv2, time
import numpy as np

# --- HSV segmentation range (tune for your lighting) ---
lowerLimitBlue = np.array([64, 0, 0], dtype=np.uint8)
upperLimitBlue = np.array([140, 255, 255], dtype=np.uint8)

# --- Init camera ---
picam2 = Picamera2()
picam2.preview_configuration.main.size = (1280, 720)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()
time.sleep(0.3)  # warm-up

try:
    while True:
        # Capture frame
        frame = picam2.capture_array()

        # Optional smoothing for a cleaner mask
        blur = cv2.blur(frame, (15, 15))

        # HSV mask
        hsv  = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lowerLimitBlue, upperLimitBlue)

        # Find largest contour
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            midx, midy = int(x + w / 2), int(y + h / 2)
            mid = (midx, midy)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, mid, 5, (255, 255, 255), -1)
            cv2.putText(frame, f"{mid}", (midx + 8, midy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "No blue region detected", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Show live view
        cv2.imshow("", mask)

        # Quit on 'q' or ESC
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q') or k == 27:
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
