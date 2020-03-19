#################################################################################################################################################
#importing files
#################################################################################################################################################
from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
from darknet import Darknet
import random 
import os
import pickle as pkl
import numpy.linalg as la
#################################################################################################################################################
#defining functions and classes
#################################################################################################################################################
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

def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
    
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
    
    return canvas

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim

def write(output, img):
    """
    The current implmentation only involves a Single Class per box. With a few modifications, the network can be trained on mulitlabel classification for each bbox
    Takes each of the BBox predictions and draws a rectangle around the detected objects
    """
    for i in range(len(output)):
        x = output[i].astype("int32")
        c1 = tuple(x[1:3])
        c2 = tuple(x[3:5])
        cls = x[-1]
        label = "{0}".format(classes[cls])
        color = (0,0,255) # Red
        centroid = np.array([0,0])
        if cls == 32: ## Sports Ball Label = 32
            img = cv2.rectangle(img, c1, c2,color, 3)
            centroid = np.array([ (c1[0] + c2[0])//2 , (c1[1] + c2[1])//2 ])
            # Since we assume only one ball per frame we take the strongest sports ball predictioon and draw the box. This helps eliminate other slight mismatches in case any
            break
    return img, centroid


classes = [
        'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 
        'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 
        'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 
        'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 
        'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
##################################################################################################################
## Srings and Tunable Variables + Thresh
##################################################################################################################
confidence = 0.5
nms_thesh = 0.4
start = 0
num_classes = 80    
bbox_attrs = 5 + num_classes
##################################################################################################################
# Load the Model
CUDA = torch.cuda.is_available()
print("CUDA :: ", CUDA)
print("Loading network.....")
assert os.path.isfile("yolov3.cfg"), 'Config File does not exist'
assert os.path.isfile("yolov3.weights"), 'Weights File don\'t does not exist. Please check the download, Link :: https://pjreddie.com/media/files/yolov3.weights'
model = Darknet("yolov3.cfg")
model.load_weights("yolov3.weights")
if CUDA:
    model.cuda()
model.eval()
print("Network successfully loaded")

model.net_info["height"] = 416
inp_dim = 416

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
t=100
print(len(a))
cap = cv2.VideoCapture('test.avi')
assert cap.isOpened(),'no video file'
x = ()
while(True):
    ret,frame = cap.read()
    print(ret)

    img, orig_im, dim = prep_image(frame, inp_dim)
    im_dim = torch.FloatTensor(dim).repeat(1,2)                        
    if CUDA:
        im_dim = im_dim.cuda()
        img = img.cuda()
    start = time.time()
    with torch.no_grad():   
        output = model(Variable(img), CUDA)
    print(output)
    # print("Time Taken for a frame", time.time() - start)
    # Non Maximal Suppression and confidence thresholding
    output = write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)
    
    im_dim = im_dim.repeat(output.size(0), 1)
    scaling_factor = torch.min(inp_dim/im_dim,1)[0].view(-1,1)
    
    output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim[:,0].view(-1,1))/2
    output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim[:,1].view(-1,1))/2
    
    output[:,1:5] /= scaling_factor

    for i in range(output.shape[0]):
        output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim[i,0])
        output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim[i,1])
    for u in range(len(output)):
        cls = output[u][-1]
        if cls == 32:
            t=u
            break 
    x=output[t][1:5].cpu().detach().numpy()
    if not x.any():
        continue
        
    xo=int((x[0]+x[2])/2)
    yo=int((x[1]+x[3])/2)
    print(xo)
    print(yo)
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
    for n in [-1]:
        incertidumbre=(xu[n]+yu[n])/2
        cv2.circle(frame,(int(xe[n]),int(ye[n])),int(incertidumbre),(255, 255, 0),1)
    for n in range(len(xp)): 
        incertidumbreP=(xpu[n]+ypu[n])/2
        cv2.circle(frame,(int(xp[n]),int(yp[n])),int(incertidumbreP),(0, 0, 255))
    if(len(listCenterY)>4):
        if((listCenterY[-5] < listCenterY[-4]) and(listCenterY[-4] <listCenterY[-3]) and (listCenterY[-3] > listCenterY[-2]) and(listCenterY[-2] > listCenterY[-1])):
            print("REBOUND")
            listCenterY=[]
            listCenterX=[]
            listpuntos=[]
            res=[]
            mu = np.array([0,0,0,0]).reshape(4,1)
            P = np.diag([100,100,100,100])**2
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows() 

