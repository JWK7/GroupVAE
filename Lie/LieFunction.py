import numpy as np
import math

def exp_lieSO3(A):
    R = np.matrix('1 0 0;0 1 0;0 0 1')
    for i in range(1,20):
        R = np.add(R,1/math.factorial(i)* A**i)
    return R

def log_lieSO3(R):
    cosTheta = (1/2)*(np.trace(R)-1)
    theta = np.arccos(cosTheta)
    X = np.transpose([R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]])
    coefficient = theta/(2*np.sin(theta))
    w =  coefficient*X

    return np.matrix([[0,-w[2],w[1]],[w[2],0,-w[0]],[-w[1],w[0],0]])

if __name__ == "__main__":
    x = np.pi
    y = np.pi/4
    z = np.pi/4

    matrix = np.matrix([[0,-z,y],[z,0,-x],[-y,x,0]])
    R = (exp_lieSO3(matrix))
    print(R)
    A = log_lieSO3(R)
    print(A)
    print(exp_lieSO3(A))
    print(x)
    print(y)
    print(z)