import cv2
import dlib


cnn_face_detector = dlib.cnn_face_detection_model_v1("model/mmod_human_face_detector.dat")

cap = cv2.VideoCapture(0)


while(True):

    # Extracting each frame (a video is a series of frames) out of the source,bool consist's of a boolen value which is true if the source what we have mentioned is valid or not
    bool, frame = cap.read()

    if(bool==False): # is it is false which means no valid source of video feed, and it breaks out the loop
        print("No source for video feed found, make sure you have a valid video source")
        break
    else:
        Gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
        faces_cnn = cnn_face_detector(Gray, 1)

        for face in faces_cnn:
            x = face.rect.left()
            y = face.rect.top()
            w = face.rect.right() - x
            h = face.rect.bottom() - y

            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)

        cv2.imshow("image" , frame)
        if(cv2.waitKey(1) & 0xFF==27):
           break

cv2.destroyAllWindows()
