import cv2
import dlib


hog_face_detector = dlib.get_frontal_face_detector()

cap = cv2.VideoCapture(0)



#Creating a infinite loop
while(True):

    # Extracting each frame (a video is a series of frames) out of the source,bool consist's of a boolen value which is true if the source what we have mentioned is valid or not
    bool, frame = cap.read()

    if(bool==False): # is it is false which means no valid source of video feed, and it breaks out the loop
        print("No source for video feed found, make sure you have a valid video source")
        break
    else:
        Gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
        faces_hog = hog_face_detector(Gray, 1)

        for face in faces_hog:
            x = face.left()
            y = face.top()
            w = face.right() - x
            h = face.bottom() - y


            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

        cv2.imshow("output_image.jpg", frame)

        if(cv2.waitKey(1) & 0xFF==27):
            break


cv2.destroyAllWindows() # Closes the window on which your face was Displaying
