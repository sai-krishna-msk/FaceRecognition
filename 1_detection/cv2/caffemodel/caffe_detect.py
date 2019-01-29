import numpy as np
import imutils
import cv2
import time

startX = 0
startY =0
endX = 0
endY = 0
img = cv2.imread("images/elon.jpg")


frame = imutils.resize(img, width=600)
(h, w) = frame.shape[:2]
modelFile = "model/res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = "model/deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

start = time.time()
blob = cv2.dnn.blobFromImage(frame , 1.0, (300, 300), [104, 117, 123], False, False)


net.setInput(blob)
detections = net.forward()
end = time.time()
print("Caffemodel : ", format(end - start, '.2f'))
bboxes = []
for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the prediction
	confidence = detections[0, 0, i, 2]
	#print(confidence)

		# filter out weak detections
	if confidence > 0.50:

			# compute the (x, y)-coordinates of the bounding box for
			# the face
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
		cv2.rectangle(img, (startX , startY), (endX, endY),
			(0, 0, 255), 2)


cv2.imwrite("output/result.jpg" ,img)
cv2.imshow("image" , img)
if(cv2.waitKey(0) & 0xFF==27):
	cv2.destroyAllWindows()
