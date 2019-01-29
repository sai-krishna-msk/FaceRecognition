import cv2
import dlib
import time



image = cv2.imread("images/test.jpg")

#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

hog_face_detector = dlib.get_frontal_face_detector()

start = time.time()

faces_hog = hog_face_detector(image, 1)

end = time.time()

print("Execution Time (in seconds) :")
print("HOG : ", format(end - start, '.2f'))

for face in faces_hog:
    x = face.left()
    y = face.top()
    w = face.right() - x
    h = face.bottom() - y


    cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)

cv2.imshow("output_image.jpg", image)
cv2.imwrite("output/output.jpg" , image)
if(cv2.waitKey(0) & 0xFF==27): # cv2,waitKey(1) it indicates refresh rate and 0xFF is the keycode for escape button ( if you press it then you will break the loop)

    cv2.destroyAllWindows() # Closes the window on which your face was Displaying
