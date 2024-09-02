from ultralytics import YOLOv10
 

model = YOLOv10('ultralytics/cfg/models/v10/yolov10s-p6.yaml').load('yolov10s.pt')
 
model.train(data='pcb_mix.yaml', epochs=50, imgsz=768, batch=32, device=0)