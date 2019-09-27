
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


def count_yes(current_list):
    current_count = 0
    for i in range(len(current_list)):
        if(current_list[i] <= -2):
            current_count += 1
    return current_count




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


train_index=[]
develop_index=[]
for i in range(len(_xtrain)):
    if random() <0.65:
        train_index.append(i)
    else:
        develop_index.append(i)

# print(len(train_index))
# print(len(develop_index))

_xtrain=np.array(_xtrain)
_ytrain=np.array(_ytrain)

X_train=_xtrain[train_index]
Y_train=_ytrain[train_index]

X_develop=_xtrain[develop_index]
Y_develop=_ytrain[develop_index]

print("X_train variables:",len(X_train))
print("Y_train variables:", len(Y_train))
print("train YES:", count_yes(Y_train))
print("train NO:", len(Y_train) - count_yes(Y_train))
print("\n")

print("X_develop variables:", len(X_develop))
print("Y_develop variables:", len(Y_develop))
print("train YES:", count_yes(Y_develop))
print("train NO:", len(Y_develop) - count_yes(Y_develop))
print("\n")

print("X_test variables:",len(X_test))
print("Y_test variables:", len(Y_test))
print("train YES:", count_yes(Y_test))
print("train NO:", len(Y_test) - count_yes(Y_test))


