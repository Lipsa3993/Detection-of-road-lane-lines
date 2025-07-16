import cv2
from lane_detection.core import process_frame

def main():
    cap = cv2.VideoCapture("data/test_video.mp4")  # Replace with 0 for webcam

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        output = process_frame(frame)
        cv2.imshow("Lane Detection", output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
