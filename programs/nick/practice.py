
# Nick Wagner
# practice

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from random import random
from sys import version


files_dir='/Users/nick/Desktop/BGMP/machine_learning/Harms_Project/files'

_xtrain=[]
_ytrain=[]

X_test=[]
Y_test=[]

count=0

with open(files_dir + '/hA6.tsv', 'r') as readFile:
    for line in readFile:
        line = line.strip().split()
        line[1] = float(line[1])
        
        if(count>=29999):
           X_test.append(line[0])
           Y_test.append(line[1])
        else:
           _xtrain.append(line[0])
           _ytrain.append(line[1])
 
        count+=1


       # if(float(line[1]) <= -2):
       #   _xtrain.append(line[0])
       #   _ytrain.append(line[1])
       # else:
       #     no.append(line)



train_index=[]
develop_index=[]
for i in range(len(_xtrain)):
    if random() <0.8:
        train_index.append(i)
    else:
        develop_index.append(i)

print(len(train_index))
print(len(develop_index))

_xtrain=np.array(_xtrain)
_ytrain=np.array(_ytrain)

X_train=_xtrain[train_index]
Y_train=_ytrain[train_index]

X_develop=_xtrain[develop_index]
Y_develop=_ytrain[develop_index]

print(len(X_train))
print(len(X_test))

