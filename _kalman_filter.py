import numpy as np

## pre/new: [x,y,ux,uy,timestamp]

def Kalman_pred(pre, new):
	p_minus = np.matrix( [ [pre[0]], [pre[1]], [pre[2]], [pre[3]] ] )
	dt = new[-1] - pre[-1]
	A = np.matrix( [ [1,0,dt,0], [0,1,0,dt], [0,0,1,0], [0,0,0,1] ] )
	B = np.matrix( [ [0.5*dt**2,0], [0,0.5*dt**2], [dt,0], [0,dt] ] )
	acc = np.matrix( [ [(new[2]-pre[2])/dt], [(new[3]-pre[3])/dt] ] )
	p1 = np.dot(A, p_minus) + np.dot(B, acc)
	return p1

    
def Kalman_corr(pre, new, P_pre, sigma):
    p_plus = np.matrix( pre )
    omicron = np.matrix( new )
    H = np.matrix( [ [1,0,0,0], [0,1,0,0] ] )
    dt = new[-1] - pre[-1]
    Q = np.matrix( [ [0.25*dt**4,0,0.5*dt**2,0], [0, 0.25*dt**4,0,0.5*dt**3], [0.5*dt**3,0,dt**2,0], [0,0.5*dt**3,0,dt**2] ] )*sigma[0]
    R = np.matrix( [ [0.25*dt**4,0,0,0], [0, 0.25*dt**4,0,0], [0,0,dt**2,0], [0,0,0,dt**2] ] )*sigma[1]
	A = np.matrix( [ [1,0,dt,0], [0,1,0,dt], [0,0,1,0], [0,0,0,1] ] )
	B = np.matrix( [ [0.5*dt**2,0], [0,0.5*dt**2], [dt,0], [0,dt] ] )
    acc = np.matrix( [ [(new[2]-pre[2])/dt], [(new[3]-pre[3])/dt] ] )
    P_minus = np.dot(A, P_pre)
    P_minus = np.dot(P_minus, np.transpose(A))
    P_minus = P_minus + Q
    inv = np.dot(H, P_minus)
    inv = np.dot(inv, np.transpose(H))+R
    inv = np.linalg.pinv(inv)
    K = np.dot(P_minus, np.transpose(H))
    K = np.dot(K, inv)
    est = np.dot(H, pre[:-1])
    error = np.dot(H, np.transpose(omicron[0,0:4])) - np.transpose(est)
    p2 = np.transpose(p_plus)[:4,0]+np.dot(K,error)
    m = np.eye(4,4) - np.dot(K,H)
    P = np.dot(m,P_minus)
    return [p2, P]
