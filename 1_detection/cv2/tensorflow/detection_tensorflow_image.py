import numpy as np
import imutils
import pickle
import cv2
import os
import time

startX = 0
startY =0
endX = 0
endY = 0
img = cv2.imread("images/test.jpg")

frame = imutils.resize(img, width=600)
(h, w) = frame.shape[:2]
model = "model/opencv_face_detector_uint8.pb"
config = "model/opencv_face_detector.pbtxt"
net =    cv2.dnn.readNetFromTensorflow(model, config)

start = time.time()
blob = cv2.dnn.blobFromImage(frame , 1.0, (300, 300), [104, 117, 123], False, False)

net.setInput(blob)
detections = net.forward()
end = time.time()
print("Tensorflow model  : ", format(end - start, '.2f'))
bboxes = []
for i in range(0, detections.shape[2]):

	confidence = detections[0, 0, i, 2]

	if confidence > 0.99:

		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
		cv2.rectangle(frame, (startX , startY), (endX, endY),
			(0, 0, 255), 2)


cv2.imwrite("output/result.jpg" ,frame)
cv2.imshow("image" , frame)
if(cv2.waitKey(0) & 0xFF==27):
		cv2.destroyAllWindows()
