from ultralytics import YOLO

def main():
    model = YOLO("yolo11n.pt")

    model.train(
        data="dataset/data.yaml",
        epochs=80,
        imgsz=640,
        batch=8,
        device=0,        
        workers=0,       
        patience=15,
        mosaic=0.0,     
        cls=0.7,         
        project="runs",
        name="plastic_train_v2"
    )

if __name__ == "__main__":
    main()
