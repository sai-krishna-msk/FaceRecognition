import cv2
import os
import pickle
import numpy as np
from PIL import Image

face_detector = cv2.CascadeClassifier("model/haarcascade_frontalface_default.xml")
names= []
images =[]
classes = []

def takeImages():
    print("How many people do you want to train your model on ??")
    no = input(": ")

    for i in range(int(no)):
        count =0
        print("PLease enter your name ")
        name = input(": ")
        names.append(name)
        if not os.path.exists("dataset/"+name):
            os.makedirs("dataset/"+name)
        source = cv2.VideoCapture(0)
        while True:
            if(count==10):
                return
            bool , img = source.read()
            Gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(Gray , 1.3  ,5)

            for(x , y , w , h) in faces:
                cv2.rectangle(img , (x , y), (x+w, y+h) , (255 , 0 , 0) , 2)
                cv2.imshow("image" , img)
                if(cv2.waitKey(1) & 0XFF==27):
                    break
                path = "dataset/"+name+"/"+str(count)+".jpg"

                cv2.imwrite(path,Gray[y:y+h,x:x+w] )
                count = count +1

    with open("model/labels.pickle" , "wb") as pickle_file:
        pickle.dump(names , pickle_file)
    cv2.destroyAllWindows()
def train():
    classes=[]
    images = []
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    for name in names:
        path = "dataset/"+name
        for each in os.listdir(path):
            each = path+"/"+each

            PIL_img = Image.open(each).convert('L') # convert it to grayscale
            currentImage = np.array(PIL_img,'uint8')
            if currentImage is None:
                print("the current image is empty")
            else:
                images.append(currentImage)
                classes.append(names.index(name))

    print(images)
    print(classes)
    classes = np.array(classes)
    recognizer.train(images , classes)
    recognizer.write('trainer/trainer.yml')
    return "Done"

takeImages()
print("Your images are ready to be trained")
train()
print("Your training process has been completed")
