import cv2
import pickle
import os


bool = 1
names=[]
with open("model/labels.pickle" , "rb") as pickle_file:
    names=pickle.load(pickle_file)

if not os.path.isfile("trainer/trainer.yml"):
    print("You have not trained your model, first kindly run train.py file")
    bool=0

if(bool==1):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')
    cascadePath = "model/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath);
    font = cv2.FONT_HERSHEY_SIMPLEX
    source = cv2.VideoCapture(0)


    while True:

        ret, img =source.read()


        Gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(Gray , 1.2 , 5)


        for(x,y,w,h) in faces:

            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

            id, confidence = recognizer.predict(Gray[y:y+h,x:x+w])


            if (confidence < 100):
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))

            cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
            cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)

        cv2.imshow('camera',img)
        if(cv2.waitKey(1)& 0XFF==27):
            break




    cv2.destroyAllWindows()
