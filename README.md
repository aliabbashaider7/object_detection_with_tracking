# Object Detection and Tracking:
A pytorch based integration of deepsort algorithm with any object detection model.

# Pre-requisits:
The code is tested on ubuntu 20.04 with python3.8 with torch >= 1.7.1 (recommended)

pip3 install -r requirements.txt
pip3 install https://github.com/KaiyangZhou/deep-person-reid/archive/master.zip

Note: Torchreid Package is not available on official PyPi hence can't be installed by simple pip command. Run the command above to install all the pre-requisits for tracker algorithm.

# Deepsort Integration:
The deepsort algorithm basically needs 3 basic elements that come from any object detection model, hence it can be worked with your object detection model as well.

Elements are:
1) Detection boxes -> [[x1,y1,w1,h1], [x2,y2,w2,h2], ...]
2) Confidences -> [[conf1], [conf2], ...]
3) Class IDs -> [[cls1], [cls2], ...]

Note that these elements are updating in every frame and these lists will length equal to the number of objects detected every frame.

In this repository, these elements are being fetched from detect_objects function in inference.py. You can customize your detection model according to get the vitals in above mentioned directory to get outputs with tracking IDs.

# Models:
Download the pretrained yolov5s.pt model from [here](https://drive.google.com/file/d/1ITEIodeXMGDgku6zN7_-_eLr9sHomoq-/view?usp=sharing) or from official yolov5 [repository](https://github.com/ultralytics/yolov5) by ultralytics. The Deepsort model will be downloaded automatically in your torch environment.

# Inference
Other than tracking model dependencies, You'll need to install all the dependencies of your own object detection algorithm of course. This integration is done with simple yolov5s object detection model. Run object_tracker.py after setting up correct video path as input in the script.


https://user-images.githubusercontent.com/84595846/162565198-ad4b5ceb-66f1-43ea-b1c1-549ff74bd0de.mp4


