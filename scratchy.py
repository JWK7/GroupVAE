import numpy as np
import math

x = np.pi/3
y = np.pi/2
z = np.pi/4

matrix = np.matrix([[0,-z,y],[z,0,-x],[-y,x,0]])
Rx = np.matrix([[1,0,0],[0,math.cos(x),-math.sin(x)],[0,math.sin(x),math.cos(x)]])
Ry = np.matrix([[math.cos(y),0,math.sin(y)],[0,1,0],[-math.sin(y),0,math.cos(y)]])
Rz = np.matrix([[math.cos(z),-math.sin(z),0],[math.sin(z),math.cos(z),0],[0,0,1]])

A3 = np.matrix([[2*(x*x-1)+1,2*x*y-2*z,2*x*y+2*y],[2*x*y+2*z,2*(y*y)+1,2*y*z-2*x*x],[2*x*z-2*y,2*y*z+2*x,2*(z*z-1)+1]])
print(A3)
# print(Rz)