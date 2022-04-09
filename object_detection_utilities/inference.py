import torch
from object_detection_utilities.datasets import LoadImages
from object_detection_utilities.general import non_max_suppression, scale_coords

def detect_objects(image, model_data):

    (model, imgsz, device) = model_data
    dataset = LoadImages(image, img_size=imgsz)

    # Get names and colors
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img) if device.type != 'cpu' else None  # run once
    detections = []
    ids = []
    confs = []
    for img, im0s in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = model(img, augment=False)[0]
        pred = non_max_suppression(pred, 0.25, 0.45)

        for i, det in enumerate(pred):  # detections per image
            if len(det):

                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

                for *xyxy, conf, cls in det:

                    # label = f'{names[int(cls)]} {conf:.2f}'
                    ids.append(int(cls))
                    confs.append(round(float(conf), 2))
                    detections.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]-xyxy[0]), int(xyxy[3]-xyxy[1])])
        return detections, ids, confs