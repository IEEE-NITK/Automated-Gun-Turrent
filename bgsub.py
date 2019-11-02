import cv2
cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = False)
while(True):
	ret,frame = cap.read()
	fgmask = fgbg.apply(frame)
	cv2.imshow('frame',fgmask)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cap.release()
cv2.destroyAllWindows() 
	

