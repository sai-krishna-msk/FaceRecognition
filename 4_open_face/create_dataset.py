from config import *
import os
import cv2
import pickle
import imutils
import numpy as np
from helper import embeddingsGenrerator
from helper import trainSVM

def CreateDataset():

    names = []
    data = []

    protoPath = os.path.sep.join("model/deploy.prototxt")
    modelPath = os.path.sep.join("model/res10_300x300_ssd_iter_140000.caffemodel")
    net= cv2.dnn.readNetFromCaffe("model/deploy.prototxt" , "model/res10_300x300_ssd_iter_140000.caffemodel")
    print("How many people you want to train the model on")
    no = input(" : ")
    if(int(no)<=1):
        if(len(os.listdir(os.getcwd()+'/dataset'))<1):
            print("You should Atleast have two classes for training !")
            return False

    for i in range(int(no)):
        count = 0
        print("Kindly Enter your name")
        name = input(" : ")
        names.append(name)
        if not os.path.exists("dataset/"+name):
            os.makedirs("dataset/"+name)

        source = cv2.VideoCapture(0)
        while True:
            if(count==TRAINING_SIZE):
            	cv2.destroyAllWindows()
            	break
            bool , frame = source.read()
            img = frame.copy()
            frame = cv2.resize(frame , (320 , 240))
            frame = imutils.resize(frame , width= 600)
            (h , w) = frame.shape[:2]
            imageBlob = cv2.dnn.blobFromImage(
        		cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        		(104.0, 177.0, 123.0), swapRB=False, crop=False)


            net.setInput(imageBlob)
            detections = net.forward()

            for j in range(0 , detections.shape[2]):
                confidence = detections[0 , 0 , j ,2]
                if confidence> 0.95:
                    box = detections[0 , 0 , j , 3:7]*np.array([w , h , w , h])
                    (startX , startY , endX , endY) = box.astype("int")

                    face = frame[startY:endY , startX:endX]

                    (fH  , fW) = face.shape[:2]

                    if fW < 20 or fH <20:
                        continue
                    cv2.rectangle(frame , (startX , startY) , (endX , endY) , (0 , 0 , 255) , 2)
                    path= "dataset/"+name+"/"+str(count)+".jpg"
                    cv2.imwrite(path , img)
                    count = count+1

            cv2.imshow("Frame" , frame)
            
            if(cv2.waitKey(1) & 0XFF==27):
                break

    return True

    
if __name__=="__main__":
    CreateDataset()