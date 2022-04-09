#!/usr/bin/env python
import cv2
import torch
import numpy as np

from models.load_models import load_object_detection_model
from object_detection_utilities.inference import detect_objects

from deep_sort.parser import get_config
from deep_sort.deep_sort import DeepSort

device = torch.device('cuda:0') # 'cpu' for cpu inference
weights_object_detection = 'yolov5s.pt'
deep_sort_model = 'osnet_x0_25'
config_deepsort = "deep_sort/configs/deep_sort.yaml"

# Load The Models
objects_model_data = load_object_detection_model(device, weights_object_detection)

class_names = objects_model_data[0].module.names if hasattr(objects_model_data[0], 'module') else objects_model_data[0].names

colors = []
for i in range(100):
    color = np.random.randint(0, 255, size=(3,))
    color = (int(color[0]), int(color[1]), int(color[2]))
    colors.append(color)

cfg = get_config()
cfg.merge_from_file(config_deepsort)
deepsort = DeepSort(deep_sort_model,
                    device,
                    max_dist=cfg.DEEPSORT.MAX_DIST,
                    max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    )

def main():
    #Test Video
    test_video = 'demo_video/road.mp4'
    vid = cv2.VideoCapture(test_video)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    #Set Output Video
    out_video = cv2.VideoWriter('out_video.mp4', codec, fps, (width, height))
    while True:
        ret, frame = vid.read()
        if ret:
            ori_height, ori_width, _ = frame.shape
            ori_frame = frame.copy()

            detections, ids, confs = detect_objects(ori_frame, objects_model_data)
            detections_tensors = torch.Tensor(detections)
            ids_tensors = torch.Tensor(ids)
            confs_tensors = torch.Tensor(confs)
            outputs = []
            if len(detections) > 0:
                outputs = deepsort.update(detections_tensors.cpu(), confs_tensors.cpu(), ids_tensors.cpu(), frame)
            for out in outputs:
                width_box = out[2]-out[0]
                height_box = out[3]-out[1]
                top_left = (out[0]+width_box//2, out[1]+height_box//2)
                bottom_right = (out[2]+width_box//2, out[3]+height_box//2)
                class_name = class_names[out[5]]
                tracking_id = out[4]
                if tracking_id>=99:
                    color = colors[-1]
                else:
                    color = colors[tracking_id]
                cv2.rectangle(ori_frame, top_left, bottom_right, color, 2)
                cv2.putText(ori_frame, f'{class_name} {tracking_id}', top_left, cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
            cv2.imshow("Output Image", ori_frame)
            cv2.waitKey(1)
            out_video.write(ori_frame)
        else:
            print('Stream Ended...')
            break
    vid.release()
    out_video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
