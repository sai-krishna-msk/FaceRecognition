
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2




modelFile = "model/res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = "model/deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

vs = VideoStream(src=0).start()
time.sleep(2)

fps = FPS().start()

while True:

	frame = vs.read()
	frame = cv2.resize(frame , (320,240))


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

			# extract the face ROI
			face = frame[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]

			if fW < 20 or fH < 20:
				continue
			cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)

	fps.update()

	cv2.imshow("Frame", frame)
	if(key = cv2.waitKey(1) & 0xFF==27):
		break




fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
cv2.destroyAllWindows()
vs.stop()
