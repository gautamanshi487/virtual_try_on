import cv2

def capture_image(filename="input.jpg"):
    cap = cv2.VideoCapture(0)
    print("Press 's' to save and exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite(filename, frame)
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_image()
