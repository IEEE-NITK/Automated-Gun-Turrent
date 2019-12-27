import cv2
import numpy as np
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

