# importing open cv module
import cv2


#importing the haarcascade model which has been already trained to detect faces
Cascade = cv2.CascadeClassifier("model/haarcascade_frontalface_default.xml")


img = cv2.imread("images/test.jpg") # Taking the image as input from images folder named test.jpg

Gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY) # an open cv funciton which converts your input imageinto balck and white( color photos take more time)
cordinates = Cascade.detectMultiScale(Gray , 1.3, 5)  #Sending the converted frame into the model , which returns the 4 cordinates points where the model expexts face is located
for (x,y,w,h) in cordinates:  # extracting the four cordinates
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)   # drawing the rectangle out of those our cordinates

cv2.imshow("image" , img) # Displaying the image on which rectabgle is drawn
cv2.imwrite("output/output.jpg" , img)


if(cv2.waitKey(0) & 0xFF==27): # cv2,waitKey(1) it indicates refresh rate and 0xFF is the keycode for escape button ( if you press it then you will break the loop)

    cv2.destroyAllWindows() # Closes the window on which your face was Displaying
