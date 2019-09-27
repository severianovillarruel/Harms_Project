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

data= []

aa_convert = {"A":0, "R":1, "N":2, "D":3, "C":4,
              "E":5, "Q":6, "G":7, "H":8, "I":9,
              "L":10, "K":11, "M":12, "F":13, "P":14,
              "S":15,"T":16, "W":17, "Y":18, "V":19}

INPUT_FILE = open("/home/severiano/harms_proj/files/hA6.tsv", "r")

for line in INPUT_FILE:
    line = line.strip().split()
    label = float(line[1])
    for i in line[0]:
        if i in aa_convert:
            data.append(aa_convert[i]/19)
    if label <= -2:
        label = 1
        _xtrain_bound.append(data) #index this list to make train, test, develop
        _ytrain_bound.append(label)
    else:
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

print(x_test[0:10])
print(x_train[0:10])

# input_layer=tf.keras.layers.Input(shape=(12,))
# nn = tf.keras.layers.Dense(25)(input_layer)
# nn = tf.keras.layers.LeakyReLU()(nn)
# nn = tf.keras.layers.Dense(25)(nn)
# nn = tf.keras.layers.LeakyReLU()(nn)
# nn = tf.keras.layers.Dense(25)(nn)
# nn = tf.keras.layers.LeakyReLU()(nn)
# output_layer = tf.keras.layers.Dense(1,activation='sigmoid')(nn)
#
# #A keras model is a way of going from one layer to the next
# wine_model=tf.keras.models.Model(input_layer,output_layer)
# wine_model.summary()
# wine_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
#
# wine_model.fit(wd.x_train,wd.y_train,epochs=1000,validation_data=(wd.x_develop,wd.y_develop)) #Have Keras make a test/validation split for us

INPUT_FILE .close()
