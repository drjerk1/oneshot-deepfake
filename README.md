# oneshot-deepfake
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
                        Input image
  --output-image OUTPUT_IMAGE
                        Output image
  --source-image SOURCE_IMAGE
                        Source image
  --refl-coef REFL_COEF
                        Reflection coeficent
  --fallback-point-detector
  
  ## python main.py - for video swapping support reading from webcam / video file and writing to webcam (using akvcam) / video file / showing in window
  ## demo example - python main.py --source-image images/1.jpg --target-image images/2.jpg --verbose
  usage: main.py [-h] [--input-video INPUT_VIDEO] [--output-video OUTPUT_VIDEO] [--output-camera OUTPUT_CAMERA] [--camera-width CAMERA_WIDTH] [--camera-height CAMERA_HEIGHT] --source-image SOURCE_IMAGE [--target-image TARGET_IMAGE]
               [--verbose] [--refl-coef REFL_COEF] [--fallback-point-detector] [--disable-faceid]

One shot face swapper

optional arguments:
  -h, --help            show this help message and exit
  --input-video INPUT_VIDEO
                        Input video
  --output-video OUTPUT_VIDEO
                        Output video
  --output-camera OUTPUT_CAMERA
                        Output camera
  --camera-width CAMERA_WIDTH
                        Output camera width
  --camera-height CAMERA_HEIGHT
                        Output camera height
  --source-image SOURCE_IMAGE
                        Source image for face swap
  --target-image TARGET_IMAGE
                        Target image for face swap
  --verbose             View progress in window
  --refl-coef REFL_COEF
                        Reflection coeficent
  --fallback-point-detector
  --disable-faceid
