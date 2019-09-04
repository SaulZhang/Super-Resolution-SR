import os
import numpy as np
path = "./subject_2"

a = np.array([1.74175371e+04,2.52134258e+04,-1.99276641e+04,1.39662295e+04,2.00832090e+04,2.68529126e+03,2.71364316e+04,3.26030020e+04,1.24374027e+01])
b = np.array([17409.06,25985.89,-19931.52,13999.92,20087.04,2678.05,27126.84,35135.29,0.])

print(np.sum((a-b)**2))

print(np.sum((a-b)**2)/len(a.flatten()))

# allFilePath = []

# for i,dir in enumerate(os.listdir(path)):
# 	list2 = os.listdir(os.path.join(path,dir))
# 	for j in list2:
# 		allFilePath.append(os.path.join(path,dir,j))

# print(len(allFilePath))
# print(allFilePath)