import os
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from model_service.pytorch_model_service import PTServingBaseService

class yolo_detection(PTServingBaseService):
# class yolo_detection():

    def __init__(self, model_name, model_path):
        print('model_name:',model_name)
        print('model_path:',model_path)
        
        path_dir = os.path.join(os.path.dirname(model_path))
        self.model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=model_path,
            config_path=os.path.join(path_dir, 'ultralytics/cfg/models/v10/yolov10-asf.yaml'),
            device='cpu'
        )
        
        self.capture = "test.jpg"
        self.score_thr = 0.3

        # 加载标签
        self.label = ['Mouse_bite', 'Open_circuit', 'Short', 'Spur', 'Spurious_copper']


    def _preprocess(self, data):
        # preprocessed_data = {}
        for _, v in data.items():
            for _, file_content in v.items():
                with open(self.capture, 'wb') as f:
                    file_content_bytes = file_content.read()
                    f.write(file_content_bytes)
        return "ok"

    def _postprocess(self, data):
        res = {
            'detection_classes': [],
            'detection_boxes': [],
            'detection_scores': []
        }

        for pred in data.object_prediction_list:
            score = pred.score.value
            if score < self.score_thr:
                continue

            bbox = pred.bbox
            cls = pred.category.id
            xmin, ymin, xmax, ymax =bbox.minx, bbox.miny, bbox.maxx, bbox.maxy

            res['detection_classes'].append(self.label[int(cls)])
            res['detection_boxes'].append([int(ymin), int(xmin), int(ymax), int(xmax)])
            res['detection_scores'].append(float(score))

        print(res)

        return res

    def _inference(self, data):
        result = get_sliced_prediction(
            self.capture,
            self.model,
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2
        )

        
        return result

# if __name__ == '__main__':
#     model_ = yolo_detection('', '/home/linj/workspace/yolov10/runs/detect/train3/weights/best.pt')
#     res = model_._inference(None)
#     model_._postprocess(res)
