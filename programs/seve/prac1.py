#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import random
from sys import version

_xtrain_bound = []
_ytrain_bound = []

_xtrain_unbound = []
_ytrain_unbound = []

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


INPUT_FILE = open("/home/severiano/harms_proj/files/hA6.tsv", "r")

for line in INPUT_FILE:
    line = line.strip().split()
    label = float(line[1])
    data = line[0]
    if label <= -2:
        label = 1
        _xtrain_bound.append(data) #index this list to make train, test, develop
        _ytrain_bound.append(label)
    if label > -2:
        label = 0
        _xtrain_unbound.append(data)
        _ytrain_unbound.append(label)
_xtrain_bound = np.array(_xtrain_bound)
_xtrain_unbound = np.array(_xtrain_unbound)

# print(len(_xtrain_bound))

test_index=random.sample(range(len(_xtrain_bound)), (len(_xtrain_bound)//4))
x_test_bound=_xtrain_bound[test_index]
x_test_unbound=_xtrain_unbound[test_index]
x_test=np.concatenate((x_test_bound,x_test_unbound), axis=0)

print(len(test_index))
for i in test_index:
    _xtrain_bound=np.delete(_xtrain_bound,i,0)
    _xtrain_unbound=np.delete(_xtrain_unbound,i,0)



INPUT_FILE .close()
