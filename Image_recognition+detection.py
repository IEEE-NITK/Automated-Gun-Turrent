import cv2
import numpy as np
from imageai.Detection import ObjectDetection
from PIL import Image
import os
execution_path = os.getcwd()
flag = 1
detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath( os.path.join(execution_path , "yolo.h5"))
detector.loadModel()
custom=detector.CustomObjects(person=True)
cap = cv2.VideoCapture(0)
while(True):
	ret,frame = cap.read()
	
	if flag is 1 :
		img1 = Image.fromarray(frame, 'RGB')
		img1.save('image1.jpg')
		detections = detector.detectCustomObjectsFromImage( custom_objects=custom,
input_image=os.path.join(execution_path , "image1.jpg"), output_image_path=os.path.join(execution_path , "image3new-custom.jpg"), minimum_percentage_probability=30)
		if detections is False:
			continue
		else :
			flag = 0
			img = Image.open('image3new-custom.jpg')
			area = img.crop(detections[0]["box_points"])
			area.save('Template', 'jpeg')
			template = cv2.imread('Template',0)
			w,h = template.shape[::-1]
 
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	res = cv2.matchTemplate(frame,template,cv2.TM_CCOEFF)	
	min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(res)
	top_left = max_loc
	bottom_right = (top_left[0]+w,top_left[1]+h)
	cv2.rectangle(frame,top_left,bottom_right,255,2)
	cv2.imshow('frame',frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cap.release()
cv2.destroyAllWindows() 
