import time
import face_recognition
import cv2






cap = cv2.VideoCapture(0)


#Creating a infinite loop
while(True):

    # Extracting each frame (a video is a series of frames) out of the source,bool consist's of a boolen value which is true if the source what we have mentioned is valid or not
    bool, frame = cap.read()

    if(bool==False): # is it is false which means no valid source of video feed, and it breaks out the loop
        print("No source for video feed found, make sure you have a valid video source")
        break
    else:
        face_locations = face_recognition.face_locations(frame)

        for top, right, bottom, left in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255 , 0), 2)


        cv2.imshow("output" , frame)
        if(cv2.waitKey(1) & 0xFF==27):
            break

cv2.destroyAllWindows()
