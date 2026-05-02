import cv2
import os

def capture_dataset():
    # Setup directories
    for label in ['good', 'defect']:
        path = os.path.join('dataset', label)
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}")

    cap = cv2.VideoCapture(0)
    print("VisionInspect AI - Dataset Capture")
    print("Press 'g' for GOOD, 'd' for DEFECT, 'q' to QUIT")

    counts = {'good': 0, 'defect': 0}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.putText(frame, f"G: {counts['good']} | D: {counts['defect']}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("VisionInspect AI - Capture", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('g'):
            counts['good'] += 1
            filename = f"dataset/good/img_{counts['good']}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Captured: {filename}")
        elif key == ord('d'):
            counts['defect'] += 1
            filename = f"dataset/defect/img_{counts['defect']}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Captured: {filename}")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_dataset()
