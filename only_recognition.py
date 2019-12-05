import cv2
import numpy as np
from imageai.Detection import ObjectDetection
from PIL import Image
import os
execution_path = os.getcwd()
flag = 1
detector = ObjectDetection()
detector .setModelTypeAsTinyYOLOv3()
detector.setModelPath( os.path.join(execution_path , "yolo-tiny.h5"))
detector.loadModel(detection_speed = "flash")
custom=detector.CustomObjects(person=True)
cap = cv2.VideoCapture(0)
while(True):
    ret,frame = cap.read()
    img1 = Image.fromarray(frame, 'RGB')
    img1.save('image1.jpg')
    detections = detector.detectCustomObjectsFromImage( custom_objects=custom,input_image=os.path.join(execution_path , "image1.jpg"), output_image_path=os.path.join(execution_path , "image3new-custom.jpg"), minimum_percentage_probability=30)
    outframe = cv2.imread('image3new-custom.jpg',0)
    cv2.imshow('outframe',outframe)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows() 
