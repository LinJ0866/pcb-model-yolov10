from ultralytics import YOLO
 

model = YOLO('ultralytics/cfg/models/v10/yolov10-asf.yaml')
 
model.train(data='pcb_mix.yaml', epochs=50, imgsz=640, batch=16, device=0)