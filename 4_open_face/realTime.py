from create_dataset import CreateDataset
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
import time
from helper import embeddingsGenrerator
from helper import trainSVM
names =""

protoPath = os.path.sep.join("model/deploy.prototxt")
modelPath = os.path.sep.join("model/res10_300x300_ssd_iter_140000.caffemodel")
detector = cv2.dnn.readNetFromCaffe("model/deploy.prototxt" , "model/res10_300x300_ssd_iter_140000.caffemodel")

embedder = cv2.dnn.readNetFromTorch("model/openface_nn4.small2.v1.t7")

def predict():
	try:
		recognizer = pickle.loads(open("output/recognizer.pickle", "rb").read())
		le = pickle.loads(open("output/le.pickle", "rb").read())
	except:
		print("You might have forgotten to train the model")
		resp = input("Would you like to train the model first ? Y/N ")
		if(resp=="Y" or resp=="y"):
			print("Creating Dataset...")
			respC = CreateDataset()
			if(not respC):
				return None

			print("Training....")
			embeddingsGenrerator()
			trainSVM()
			print("Training Completed !")
			recognizer = pickle.loads(open("output/recognizer.pickle", "rb").read())
			le = pickle.loads(open("output/le.pickle", "rb").read())





	cap = cv2.VideoCapture(0)


	while(True):




		ret , image = cap.read()
		image = imutils.resize(image, width=600)

		(h, w) = image.shape[:2]

		imageBlob = cv2.dnn.blobFromImage(
			cv2.resize(image, (300, 300)), 1.0, (300, 300),
			(104.0, 177.0, 123.0), swapRB=False, crop=False)

		start = time.time()
		detector.setInput(imageBlob)
		detections = detector.forward()


		for i in range(0, detections.shape[2]):
			
			confidence = detections[0, 0, i, 2]

			# filter out weak detections
			if confidence > 0.95:
				# compute the (x, y)-coordinates of the bounding box for the
				# face
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				# extract the face ROI
				face = image[startY:endY, startX:endX]
				(fH, fW) = face.shape[:2]

				# ensure the face width and height are sufficiently large
				if fW < 20 or fH < 20:
					continue

				# construct a blob for the face ROI, then pass the blob
				# through our face embedding model to obtain the 128-d
				# quantification of the face
				faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
					(0, 0, 0), swapRB=True, crop=False)
				embedder.setInput(faceBlob)
				vec = embedder.forward()

				# perform classification to recognize the face
				preds = recognizer.predict_proba(vec)[0]
				j = np.argmax(preds)
				proba = preds[j]
				name = le.classes_[j]
				names = name
				y = startY - 10 if startY - 10 > 10 else startY + 10
				cv2.rectangle(image, (startX, startY), (endX, endY),
						(0, 0, 255), 2)
				cv2.putText(image, names, (startX, y),
						cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
		cv2.imshow("Frame", image)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

	cv2.destroyAllWindows()

if __name__=="__main__":
	resp =predict()
