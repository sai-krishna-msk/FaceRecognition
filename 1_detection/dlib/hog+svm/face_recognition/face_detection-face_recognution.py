import time
import face_recognition
import cv2

image = cv2.imread("images/test.jpg")
start = time.time()
face_locations = face_recognition.face_locations(image)
end = time.time()


print("Execution Time (in seconds) :")
print("face_recognition-module : ", format(end - start, '.2f'))

for top, right, bottom, left in face_locations:
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255 , 0), 2)

cv2.imwrite("output_image-face_recognition_module.jpg", image)

cv2.imshow("output" , image)
if(cv2.waitKey(0) & 0xFF==27):
    cv2.destroyAllWindows()
