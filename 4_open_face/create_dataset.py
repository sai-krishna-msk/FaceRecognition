import os
import cv2
import pickle
import imutils
import numpy as np

names = []
data = []

modelFile = "model/res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = "model/deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
print("How many people you want to train the model on")
no = input(" : ")

for i in range(int(no)):
    count = 0
    print("Could you please enter your name")
    name = input(" : ")
    names.append(name)
    if not os.path.exists("dataset/"+name):
        os.makedirs("dataset/"+name)

    source = cv2.VideoCapture(0)
    while True:
        if(count==26):
            break
        bool , frame = source.read()
        frame = cv2.resize(frame , (320 , 240))
        frame = imutils.resize(frame , width= 600)
        (h , w) = frame.shape[:2]
        print("till ehre no pr")
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

        cv2.imshow("Frame" , frame)
        path= "dataset/"+name+"/"+str(count)+".jpg"
        cv2.imwrite(path , frame)
        count = count+1
        if(cv2.waitKey(1) & 0XFF==27):
            break

cv2.destroyAllWindows()

'''
for name in names:
    for each in os.listdir("dataset/"+name):
        image = cv2.imread("dataset/"+name+"/"+each)
        data.append([image , name])






with open("training_data.pickle", 'wb') as pickle_file:

    pickle.dump(data, pickle_file)
'''
