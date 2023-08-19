import numpy as np
import math

# class GroupData:
#     def __init__(self,Rotation,Translation):
#         if len(Rotation) != len(Translation):
#             raise KeyError("Value Not Possible")
#         self.Matrix = numpy 
#         return 0

def SO2(information):
    if information != 1:
        raise ValueError
    return np.matrix([[math.cos(information[0]),math.sin(information[0])],[-math.sin(information[0]),math.cos(information[0])]])

def SO3(information):
    if information != 3:
        raise ValueError
    return np.matrix()

def GroupCreate(groupName,information):
    if groupName == 'SO2':
        return 0
    elif groupName == 'SO3':
        return 0
    elif groupName == 'SE2':
        return 0
    elif groupName == 'SE3':
        return 0
    else:
        raise KeyError("Not a Possible Group")
    
    