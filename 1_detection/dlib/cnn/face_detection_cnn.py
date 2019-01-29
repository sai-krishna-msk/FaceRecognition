import cv2
import dlib
import time



cnn_face_detector = dlib.cnn_face_detection_model_v1("model/mmod_human_face_detector.dat")

image = cv2.imread("images/elon.jpg")

start = time.time()
faces_cnn = cnn_face_detector(image, 1)
end = time.time()
print("CNN : ", format(end - start, '.2f'))


for face in faces_cnn:
    x = face.rect.left()
    y = face.rect.top()
    w = face.rect.right() - x
    h = face.rect.bottom() - y


    cv2.rectangle(image, (x,y), (x+w,y+h), (0,0,255), 2)

cv2.imwrite("outputs/result.jpg", image)
cv2.imshow("image" , image)
if(cv2.waitKey(0) & 0xFF==27):
    cv2.destroyAllWindows()
