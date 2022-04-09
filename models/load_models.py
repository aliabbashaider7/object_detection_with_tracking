from models.experimental import attempt_load
from object_detection_utilities.general import check_img_size

def load_object_detection_model(device, weights_path):

    model = attempt_load(weights_path, map_location=device)  # load FP32 model
    imgsz = check_img_size(640, s=model.stride.max())  # check img_size

    return (model, imgsz, device)