from ultralytics import YOLO
 

model = YOLO('ultralytics/cfg/models/v10/yolov10-asf.yaml') \
    .load('/home/linj/workspace/yolov10/runs/detect/train7/weights/best.pt')
 
model.train(data='pcb_huawei.yaml', epochs=30, imgsz=640, batch=16, device=0)