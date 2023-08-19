import numpy as np

# class AlgebraData:
#     def __init__(self,Rotation,Translation):
#         if len(Rotation) != len(Translation):
#             raise KeyError("Value Not Possible")
        
#         self.Matrix = np.matrix([[0,-Rotation[0]],[Rotation[0],0]]) 
#         return 0

def so2(information):
    if len(information) != 1:
        raise ValueError
    return [[0,-information[0]],[information[0],0]]

def so3(information):
    if len(information) != 3:
        raise ValueError
    return [[0,-information[2],information[1]],[information[2],0,information[0]],[-information[1],information[0],0]]

def se2(information):
    if len(information) != 3:
        raise ValueError
    return [[0,-information[0],information[1]],[information[0],0,information[2]]]
    
def se3(information):
    if len(information) != 6:
        raise ValueError
    return [[0,-information[2],information[1],information[3]],[information[2],0,information[0],information[4]],[-information[1],information[0],0,information[5]]]

def AlgebraCreate(algName,information):
    if algName == 'so2':
        return so2(information)
    elif algName == 'so3':
        return so3(information)
    elif algName == 'se2':
        return se2(information)
    elif algName == 'se3':
        return se3(information)
    else:
        raise KeyError("Not a Possible Group")
