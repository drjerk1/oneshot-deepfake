# oneshot-deepfake

# That's is course work done by Denis Tsyupa, Vasiliy Kozlov, Vadim Chernishov at HSE University

Swap faces in the video or images in realtime using only one photo of both people

## Download weights here and place them into weights directory, in the same directory where main.py is

https://www.kaggle.com/drjerk/one-shot-deepfake-weights

## https://www.kaggle.com/drjerk/swap-faces-usage-demo - simple usage demo

## python process_image.py - for image swapping

usage: process_image.py [-h] --input-image INPUT_IMAGE --output-image OUTPUT_IMAGE --source-image SOURCE_IMAGE [--refl-coef REFL_COEF] [--fallback-point-detector]

One shot face swapper

optional arguments:

  -h, --help            show this help message and exit
  
  --input-image INPUT_IMAGE
                        
  --output-image OUTPUT_IMAGE
                        
  --source-image SOURCE_IMAGE
                        
  --refl-coef REFL_COEF
                        
  --fallback-point-detector
  
  ## python main.py - for video swapping support reading from webcam / video file and writing to webcam (using akvcam) / video file / showing in window
  
  ## demo example - python main.py --source-image images/1.jpg --target-image images/2.jpg --verbose
  
  usage: main.py [-h] [--input-video INPUT_VIDEO] [--output-video OUTPUT_VIDEO] [--output-camera OUTPUT_CAMERA] [--camera-width CAMERA_WIDTH] [--camera-height CAMERA_HEIGHT] --source-image SOURCE_IMAGE [--target-image TARGET_IMAGE]
               [--verbose] [--refl-coef REFL_COEF] [--fallback-point-detector] [--disable-faceid]

One shot face swapper

optional arguments:
  -h, --help            show this help message and exit
  --input-video INPUT_VIDEO
  
  --output-video OUTPUT_VIDEO

  --output-camera OUTPUT_CAMERA

  --camera-width CAMERA_WIDTH

  --camera-height CAMERA_HEIGHT

  --source-image SOURCE_IMAGE

  --target-image TARGET_IMAGE

  --verbose (show simple gui demo window)
  
  --refl-coef REFL_COEF

  --fallback-point-detector
  
  --disable-faceid

## Some swapping results
![demos/1.png](demos/1.png?raw=true "demos/1.png")
![demos/2.png](demos/2.png?raw=true "demos/2.png")
![demos/3.png](demos/3.png?raw=true "demos/3.png")
![demos/4.png](demos/4.png?raw=true "demos/4.png")

## Sources
https://github.com/TreB1eN/InsightFace_Pytorch - face comparison model

https://github.com/1adrianb/face-alignment - facial landmarks model used to pseudolabel dataset

https://github.com/webcamoid/akvcam - virtual camera integration

https://www.kaggle.com/yukia18/sub-rals-biggan-with-auxiliary-classifier - good biggan implementation
