from ultralytics import YOLO
import cv2

model = YOLO("weights/best.pt")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("Webcam tidak bisa dibuka")
    exit()

print("Webcam aktif | tekan Q untuk keluar")

while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        print("Frame kosong")
        continue

    results = model(frame, conf=0.6, device=0)
    annotated = results[0].plot()

    cv2.imshow("Plastic Waste Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
