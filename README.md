# Computer Pointer Controller

This project use 4 different machine learning models to control the Mouse through the see of a person. Also we can see the different times using diferent devices and configurations.

* [Face Detection Model] (https://docs.openvinotoolkit.org/2019_R1/_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html) 
* [Landmark Detection Model] (https://docs.openvinotoolkit.org/2018_R5/_docs_Retail_object_attributes_landmarks_regression_0009_onnx_desc_landmarks_regression_retail_0009.html)

* [Head Pose Detection Model] (https://docs.openvinotoolkit.org/2019_R1/_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
* [Gaze Pose Detection Model] (https://docs.openvinotoolkit.org/2019_R1/_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)


You can follow the link to check  [video] (https://drive.google.com/file/d/1An82JSIWOi2mqK3pSY7bUcyn-V12j3BY/view?usp=sharing)

![MouseComputerController](/images/MouseComputerControllerVideo.gif)

Pipeline:

As you see in the graph we receive a CAM or a video file and  use face detection model to recognize the face and we send the cropped faces to land mark that detect the eyes and head pose estimation model for detect the angles based in the head and final we use Gaze estimation for  control the mouse

![pipeline](/images/pipeline.png)

## Project Set Up and Installation

1. First you need to install and configurate Intel Open Vino toolkit. You can follow this [OpenVino](https://docs.openvinotoolkit.org/latest/) 
Be sure after installing , try Runing demo.
cd C:\Program Files (x86)\IntelSWTools\openvino\bin\
setupvars.bat
```
cd C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\demo\
demo_squeezenet_download_convert_run.bat -d MYRIAD  
```
```
demo_security_barrier_camera.bat
```

Take notice that demo is just the OpenVinotoolkit demo to be sure everything is working before running the demo for the project.

2. install all the dependency using `pip install requirements.txt`.

3. Initialize your openVINO Enviroment 

For Windows.
```
cd C:\Program Files (x86)\IntelSWTools\openvino\bin\
setupvars.bat
```

For Linux Configuration:

Initialize the openVINO environment:-
```
source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5
```

4. Downloading the Models needed.

Windows:

* Face Detection Model ADAS (Not Binary) I added this model to test with NCS2 (MYRIAD) 
```
python downloader.py --name "face-detection-adas-0001" --precisions=FP16,FP32,FP32-INT1  --output_dir c:\models
```
* Face Detection Model Binary (This model doesn't have FP16 to run MYRIAD)

```
python "<pathopenvino>/deployment_tools/tools/model_downloader/downloader.py" --name "face-detection-adas-binary-0001"
```


* Landmarks Regression Retail
```
python "<pathopenvino>/deployment_tools/tools/model_downloader/downloader.py" --name "landmarks-regression-retail-0009"
```
* Head Pose Estimation Adas
```
python "<pathopenvino>/deployment_tools/tools/model_downloader/downloader.py" --name "head-pose-estimation-adas-0001"
```


Sample:
```
c:\Program Files (x86)\IntelSWTools\openvino_2020.3.194\deployment_tools\tools\model_downloader\downloader.py --name "face-detection-adas-binary-0001"--precisions=INT8,FP16,FP32,FP32-INT1  --output_dir c:\models
```

Linux:

* Face Detection Model
```
python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "face-detection-adas-binary-0001" --output_dir ~/openvino_models/
```

* Landmarks Regression Retail
```
python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "landmarks-regression-retail-0009" --output_dir ~/openvino_models/
```

* Head Pose Estimation Adas
```
python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "head-pose-estimation-adas-0001" --output_dir ~/openvino_models/
```

* Gaze Estimation
```
python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "gaze-estimation-adas-0002" --output_dir ~/openvino_models/
```

## Demo
*  For running basic model for FP32 follow the next command.

```
cd C:\Program Files (x86)\IntelSWTools\openvino\bin\
setupvars.bat
```

```
cd c:\Projects\Udacity-AI-Intel-Computer-Pointer-Controller\src
```
```
python main.py -fm "c:\models\intel\face-detection-adas-binary-0001\FP32-INT1\face-detection-adas-binary-0001" -lm "c:\models\intel\landmarks-regression-retail-0009\FP32\landmarks-regression-retail-0009" -hm "c:\models\intel\head-pose-estimation-adas-0001\FP32\head-pose-estimation-adas-0001" -gm "c:\models\intel\gaze-estimation-adas-0002\FP32\gaze-estimation-adas-0002"  -i c:\Projects\Udacity-AI-Intel-Computer-Pointer-Controller\bin\demo.mp4 -d CPU,CPU,CPU,CPU -o benchmark/CPU/FP32/ 
```

## Documentation
### Arguments Documentation 

Following are commanda line arguments that can use for while running the main.py file ` python main.py `:-
```
  -h, --help            show this help message and exit
  -fm FACE_DETECTION_MODEL, --face_detection_model FACE_DETECTION_MODEL
                        Path xml file - face detection model
  -lm FACIAL_LANDMARKS_MODEL, --facial_landmarks_model FACIAL_LANDMARKS_MODEL
                        Path xml file - facial landmarks detection model
  -hm HEAD_POSE_MODEL, --head_pose_model HEAD_POSE_MODEL
                        Path xml file head pose estimation model xml
  -gm GAZE_ESTIMATION_MODEL, --gaze_estimation_model GAZE_ESTIMATION_MODEL
                        Path xml file gaze estimation model
  -i INPUT, --input INPUT
                        Path video file or CAM
  -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION
                        MKLDNN (CPU)-targeted custom layers.Absolute path to a
                        shared library with thekernels impl.
  -d DEVICE, --device DEVICE
                        Specify the target device to infer in the order: FACE
                        DETECTION, LANDMARK, HEAD POSE AND GAZECPU, GPU, FPGA
                        or MYRIAD is acceptable. Sample (CPU CPU CPU CPU by
                        default)MYRIAD SAMPLE: CPU,MYRIAD,MYRIAD,MYRIAD
  -pt PROB_THRESHOLD, --prob_threshold PROB_THRESHOLD
                        Probability threshold for detections filtering(0.5 by
                        default)
  -vf VISUALIZATION_FLAG [VISUALIZATION_FLAG ...], --visualization_flag VISUALIZATION_FLAG [VISUALIZATION_FLAG ...]      
                        Example: for see the visualization of different model
                        outputs of each frame,fm for Face Detection Model, lm
                        for Facial Landmark Detection Modelhm for Head Pose
                        Estimation Model, gm for Gaze Estimation Model.
  -o OUTPUT_PATH, --output_path OUTPUT_PATH.
```

## Benchmarks


For running  and testing models I'm using the following hardware and SO characteristics.
```
OS Name:                   Microsoft Windows 10 Enterprise
OS Version:                10.0.18362 N/A Build 18362
OS Build Type:             Multiprocessor 
System Type:               x64-based PC
Memory                     32 GB Memory
Processor(s):              i7-9750H CPU  6 Cores GenuineIntel ~2592 Mhz

```
Also I test a  Intel neural Compute Stick2 
![NCS2](/images/NCS2.jpeg?raw=true "NCS2")


Information of the different metrics  `Inference`, `Frames per Second` and `Load Model Time` by the precisions (FP16, FP16-INT8,FP32)

### CPU

#### FP32:
```
python main.py -fm "c:\models\intel\face-detection-adas-binary-0001\FP32-INT1\face-detection-adas-binary-0001" -lm "c:\models\intel\landmarks-regression-retail-0009\FP32\landmarks-regression-retail-0009" -hm "c:\models\intel\head-pose-estimation-adas-0001\FP32\head-pose-estimation-adas-0001" -gm "c:\models\intel\gaze-estimation-adas-0002\FP32\gaze-estimation-adas-0002"  -i c:\Projects\Udacity-AI-Intel-Computer-Pointer-Controller\bin\demo.mp4 -d CPU,CPU,CPU,CPU -o benchmark/CPU/FP32/ -vf fm lm hm gm
```

#### FP16-INT8:
```
python main.py -fm "c:\models\intel\face-detection-adas-binary-0001\FP32-INT1\face-detection-adas-binary-0001" -lm "c:\models\intel\landmarks-regression-retail-0009\FP16-INT8\landmarks-regression-retail-0009" -hm "c:\models\intel\head-pose-estimation-adas-0001\FP16-INT8\head-pose-estimation-adas-0001" -gm "c:\models\intel\gaze-estimation-adas-0002\FP16-INT8\gaze-estimation-adas-0002"  -i c:\Projects\Udacity-AI-Intel-Computer-Pointer-Controller\bin\demo.mp4 -d CPU,CPU,CPU,CPU -o benchmark/CPU/FP16-INT8/ -vf fm lm hm gm
```

#### FP16
```
python main.py -fm "c:\models\intel\face-detection-adas-0001\FP16\face-detection-adas-0001" -lm "c:\models\intel\landmarks-regression-retail-0009\FP16\landmarks-regression-retail-0009" -hm "c:\models\intel\head-pose-estimation-adas-0001\FP16\head-pose-estimation-adas-0001" -gm "c:\models\intel\gaze-estimation-adas-0002\FP16\gaze-estimation-adas-0002"  -i c:\Projects\Udacity-AI-Intel-Computer-Pointer-Controller\bin\demo.mp4 -d CPU,CPU,CPU,CPU -o benchmark/CPU/FP16/ -vf fm lm hm gm
```


### MYRAD

### FP16:

python main.py -fm "c:\models\intel\face-detection-adas-0001\FP16\face-detection-adas-0001" -lm "c:\models\intel\landmarks-regression-retail-0009\FP16\landmarks-regression-retail-0009" -hm "c:\models\intel\head-pose-estimation-adas-0001\FP16\head-pose-estimation-adas-0001" -gm "c:\models\intel\gaze-estimation-adas-0002\FP16\gaze-estimation-adas-0002"  -i c:\Projects\Udacity-AI-Intel-Computer-Pointer-Controller\bin\demo.mp4 -d MYRIAD,MYRIAD,MYRIAD,MYRIAD -o benchmark/MYRIAD/FP16/ -vf fm lm hm gm

### Total Metrics

#### Total Metrics by precision
![Total Metrics by Precision](/images/TotalMetricsPrecision.png)

#### Total Inference tme  by precision
![Total Inference by Precision](/images/TotalInferencetime.png)

#### Total Frames per Second  by precision
![Total FPS by Precision](/images/TotalFPS.png)

#### Total Model load time by precision
![Total Model Load time by Precision](/images/TotalModelload.png)

### Model Metrics

####  All Models comparison - load time by precision

![Face Model load time by precision](/images/Loadtimeprecision.png)

#### Face Detection Model - load time by precision
![Face Model load time by precision](/images/FacePrecision.png)

#### Land Mark Detection Model - load time by precision
![Land Mark Detection Model](/images/LandPrecision.png)

#### Head Pose estimation Model - load time by precision
![Head Pose Detection Model](/images/HeadPrecision.png)

#### Gaze estimation Model - load time by precision
![Gaze estimation Model](/images/GazePrecision.png)


## Results

* We can see based in the metrics that FP32 and FP16-INT8 is more performance than  FP16 for the Inference time Metric and also for Frame Per Second metric.  Also we can see that Model Load time FP32 and FP16 takes less time loading the model than FP16-INT8

* We can see based in the metrics that FP32 and FP16-INT8 is more performance than  FP16 for the Inference time Metric and also for Frame Per Second metric.  Also we can see that Model Load time FP32 and FP16 takes less time loading the model than FP16-INT8

* We can see that a type of precision is not necessarily going to be better for the different models, this depends on each model.

* FP32 and FP16-INT8 are slightly better in loading time than the face detection model.

* FP16 are slightly better at load time than Landmark model

* FP32 is significantly better in head pose model load time than FP16 and much better than FP16-INT8.

* FP32 is significantly better in loading time than the Gaze model than FP16 and much better than FP16-INT8.


## Stand Out Suggestions
Other Result is the MYRIAD (NCS2) I tried to run over the LANDMARK, HEADPOSE and GAZE because FACE model doesn't support FP16 but even for them I'm getting a error seems to be a problem reported in Intel for other users. The extrange is that NCS2 is working for the demo but not in this case.

### Async Inference

I also tested models for Synchronous,Ansynchronous and the last one is slightly better. 


### Edge Cases

When you have more that a face detection you skip it to just use the first one.

I notice that when you get the corners for control the mouse  the library gets a safe out error so you can use pyautogui.FAILSAFE = False 
