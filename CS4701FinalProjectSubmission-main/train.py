from ultralytics import YOLO

model = YOLO("yolov8m.pt")
model.train(data="data.yaml", epochs=25)

# model = YOLO("runs/detect/train3/weights/best.pt")
# model.train(resume=True)