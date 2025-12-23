from ultralytics import YOLO
import cv2

model = YOLO("weights/best.pt")

img_path = "D:\\kuliah\\materi\\viskom\\plastic-waste-detection\\dataset\\test\\images\\WhatsApp-Image-2025-12-23-at-02-13-20_jpeg.rf.91219e484b95684ed605991b74a6688f.jpg"

results = model(img_path, conf=0.4, device=0)
annotated = results[0].plot()

cv2.imshow("Image Detection", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
