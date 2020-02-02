import cv2
import numpy as np
from imageai.Detection import ObjectDetection
from PIL import Image
import os

import numpy.linalg as la

def kalman(mu,P,F,Q,B,u,z,H,R):
    # mu, P : current state and its uncertainty
    # F, Q  : Dynamic system and its noise
    # B, u  : control model and the entrance
    # z     : observation
    # H, R  : Observation model and its noise

    mup = F @ mu + B @ u;
    pp  = F @ P @ F.T + Q;

    zp = H @ mup

    # if there is no observation we only do prediction

    if z is None:
        return mup, pp, zp

    epsilon = z - zp

    k = pp @ H.T @ la.inv(H @ pp @ H.T +R)

    new_mu = mup + k @ epsilon;
    new_P  = (np.eye(len(P))-k @ H) @ pp;
    return new_mu, new_P, zp

fps=30
dt=1/fps
t = np.arange(0,2.01,dt)
noise = 3

degree = np.pi/180
a = np.array([0, 900])

F=np.array([1, 0, dt, 0,
0, 1, 0, dt,
0, 0, 1, 0,
0, 0, 0, 1]).reshape(4,4)

B=np.array([dt**2/2, 0,
0, dt**2/2,
dt, 0,
0, dt]).reshape(4,2)

H = np.array(
[1,0,0,0,
0,1,0,0]).reshape(2,4)

P = np.diag([1000,1000,1000,1000])**2

mu = np.array([0,0,0,0])

res=[]
N = 15

sigmaM = 0.0001
sigmaZ = 3*noise

Q = sigmaM**2 * np.eye(4)
R = sigmaZ**2 * np.eye(2)

listCenterX=[]
listCenterY=[]
listpuntos=[]

execution_path = os.getcwd()
detector = ObjectDetection()
detector .setModelTypeAsTinyYOLOv3()
detector.setModelPath( os.path.join(execution_path , "yolo-tiny.h5"))
detector.loadModel(detection_speed = "flash")
custom=detector.CustomObjects(sports_ball=True)
cap = cv2.VideoCapture(0)
x = ()
while(True):
    ret,frame = cap.read()
    img1 = Image.fromarray(frame, 'RGB')
    img1.save('image1.jpg')
    detections,extracted_objects_array = detector.detectCustomObjectsFromImage( custom_objects=custom,input_image=os.path.join(execution_path , "image1.jpg"), output_image_path=os.path.join(execution_path , "image3new-custom.jpg"), extract_detected_objects=True, minimum_percentage_probability=30)
    for detection, object_path in zip(detections, extracted_objects_array):
        print(object_path)
        print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])
        print("---------------")
        x=detection["box_points"]
    if not x:            
        continue
    xo=int((x[0]+x[2])/2)
    yo=int((x[1]+x[3])/2)
    mu,P,pred= kalman(mu,P,F,Q,B,a,np.array([xo,yo]),H,R)
    listCenterX.append(xo)
    listCenterY.append(yo)
    res += [(mu,P)]
    mu2 = mu
    P2 = P
    res2 = []
    for _ in range(fps*2):
        mu2,P2,pred2= kalman(mu2,P2,F,Q,B,a,None,H,R)
        res2 += [(mu2,P2)]
    xe=[mu[0] for mu,_ in res]
    xu=[2*np.sqrt(P[0,0]) for _,P in res]
    ye=[mu[1] for mu,_ in res]
    yu=[2*np.sqrt(P[1,1]) for _,P in res] 
    xp=[mu2[0] for mu2,_ in res2]
    yp=[mu2[1] for mu2,_ in res2]
    xpu = [2*np.sqrt(P[0,0]) for _,P in res2]
    ypu = [2*np.sqrt(P[1,1]) for _,P in res2]
    for n in range(len(listCenterX)): 
        cv2.circle(frame,(int(listCenterX[n]),int(listCenterY[n])),3,(0, 255, 0),-1)
    incertidumbre=(xu[-1]+yu[-1])/2
    cv2.circle(frame,(int(xe[-1]),int(ye[-1])),int(incertidumbre),(255, 255, 0),1)
    for n in range(len(xp)): 
        incertidumbreP=(xpu[n]+ypu[n])/2
        cv2.circle(frame,(int(xp[n]),int(yp[n])),int(incertidumbreP),(0, 0, 255))
    listCenterY=[]
    listCenterX=[]
    listpuntos=[]
    res=[]
    mu = np.array([0,0,0,0])
    P = np.diag([100,100,100,100])**2
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break
cap.release()
cv2.destroyAllWindows() 

