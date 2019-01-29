
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os

modelFile = "model/opencv_face_detector_uint8.pb"
configFile = "model/opencv_face_detector.pbtxt"
net =    cv2.dnn.readNetFromTensorflow(modelFile, configFile)

vs = VideoStream(src=0).start()
time.sleep(2.0)


fps = FPS().start()

while True:

	frame = vs.read()


	frame = imutils.resize(frame, width=600)
	(h, w) = frame.shape[:2]

	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(frame, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

	net.setInput(imageBlob)
	detections = net.forward()

	for i in range(0, detections.shape[2]):

		confidence = detections[0, 0, i, 2]
		if confidence > 0.95:

			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			face = frame[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]

			if fW < 20 or fH < 20:
				continue
			cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)

	fps.update()

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()
