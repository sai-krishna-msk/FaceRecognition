For implementing-:

Installing the required dependencies-:
> Open the command prompt
> navigate to folder (inside the folder)
> enter-:
```
pip install opencv-contrib-python
conda install -c menpo dlib
```


Detecting you face in relative through video feed from you webcam -:

```
python face_detection_cnn_video.py
```


Detecting face in an image ( already an image in present in the images folder)
```
python face_detection_cnn.py
```
if you want your face detected for a your image-:
a) place your image , in the images folder with names as test
b) or replace the word test with you image file name placed in the images folder , in the 9th line of code of face_detect_image.py file
