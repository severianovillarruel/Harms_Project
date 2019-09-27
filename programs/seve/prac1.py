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
_ytrain_bound = np.array(_ytrain_bound)
_ytrain_unbound = np.array(_ytrain_unbound)

test_index=random.sample(range(len(_xtrain_bound)), (len(_xtrain_bound)//4))
x_test_bound=_xtrain_bound[test_index]
x_test_unbound=_xtrain_unbound[test_index]
y_test_bound=_ytrain_bound[test_index]
y_test_unbound=_ytrain_unbound[test_index]
x_test=np.concatenate((x_test_bound,x_test_unbound), axis=0)
y_test=np.concatenate((y_test_bound,y_test_unbound), axis=0)


for i in test_index:
    x_train_bound=np.delete(_xtrain_bound,i,0)
    x_train_unbound=np.delete(_xtrain_unbound,i,0)
    y_train_bound=np.delete(_ytrain_bound,i,0)
    y_train_unbound=np.delete(_ytrain_unbound,i,0)
x_train=np.concatenate((x_train_bound,x_train_unbound), axis=0)
y_train=np.concatenate((y_train_bound,y_train_unbound), axis=0)

def count_yes(current_list):
    current_count = 0
    for i in range(len(current_list)):
        if(current_list[i] == 1):
            current_count += 1
    return current_count

print("test YES:", count_yes(y_test))
print("test NO:", len(y_test) - count_yes(y_test))
print("train YES:", count_yes(y_train))
print("train NO:", len(y_train) - count_yes(y_train))
INPUT_FILE .close()
