# importing open cv module
import cv2


#importing the haarcascade model which has been already trained to detect faces
Cascade = cv2.CascadeClassifier("model/haarcascade_frontalface_default.xml")


#Creating the source of the of video input, 0 means default source of input which is the webcam of the device you are using you can also give an ip adress of an external video camera if you would like to
cap = cv2.VideoCapture(0)



#Creating a infinite loop
while(True):

    # Extracting each frame (a video is a series of frames) out of the source,bool consist's of a boolen value which is true if the source what we have mentioned is valid or not
    bool, frame = cap.read()

    if(bool==False): # is it is false which means no valid source of video feed, and it breaks out the loop
        print("No source for video feed found, make sure you have a valid video source")
        break
    else:
        Gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY) # an open cv funciton which converts your input image (a frame) you have extracted into balck and white( color photos take more time)
        cordinates = Cascade.detectMultiScale(Gray , 1.3, 5)  #Sending the converted frame into the model , which returns the 4 cordinates points where the model expexts face is located
        for (x,y,w,h) in cordinates:  # extracting the four cordinates
             cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)   # drawing the rectangle out of those our cordinates

        cv2.imshow("image" , frame) # Displaying the image on which rectabgle is drawn


        if(cv2.waitKey(1) & 0xFF==27): # cv2,waitKey(1) it indicates refresh rate and 0xFF is the keycode for escape button ( if you press it then you will break the loop)
            break



cv2.destroyAllWindows() # Closes the window on which your face was Displaying
