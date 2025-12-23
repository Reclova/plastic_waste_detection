from ultralytics import YOLO
import cv2
import os

MODEL_PATH = "weights/best.pt"
VIDEO_PATH = "videos/test_video.mp4"
OUTPUT_PATH = "output_video.mp4"
CONF_THRES = 0.6
DEVICE = 0  # GPU


model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Gagal membuka video")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

print("Memproses video... tekan Q untuk berhenti")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=CONF_THRES, device=DEVICE)
    annotated = results[0].plot()

    out.write(annotated)
    cv2.imshow("Plastic Waste Detection - Video", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video selesai diproses, hasil disimpan ke: {OUTPUT_PATH}")
