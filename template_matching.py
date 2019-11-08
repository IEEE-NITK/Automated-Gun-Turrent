import cv2
import numpy as np
cap = cv2.VideoCapture(0)
template = cv2.imread('Template.jpg',0)
w,h = template.shape[::-1]
while(True):
	ret,frame = cap.read()
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
