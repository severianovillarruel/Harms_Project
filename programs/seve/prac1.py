#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import random
from sys import version
from keras import models
from keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

_xtrain_bound = []
_ytrain_bound = []

_xtrain_unbound = []
_ytrain_unbound = []

aa_convert = {"A":0, "R":1, "N":2, "D":3, "C":4,
              "E":5, "Q":6, "G":7, "H":8, "I":9,
              "L":10, "K":11, "M":12, "F":13, "P":14,
              "S":15,"T":16, "W":17, "Y":18, "V":19}

INPUT_FILE = open("/home/severiano/harms_proj/files/hA6.tsv", "r")

for line in INPUT_FILE:
    line = line.strip().split()
    label = float(line[1])
    data=[]
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

#MAKE A LIST OF RANDOM NUMBERS FOR INDEXES
#need to learn how to make two lists of random numbers
#then to use all of these numbers to delete the index from original data
test_develop_index=random.sample(range(len(_xtrain_bound)), (len(_xtrain_bound)//4))
test_index=test_develop_index[:len(test_develop_index)//2]
develop_index=test_develop_index[len(test_develop_index)//2:]
#DELETE INDEXES FROM ORIGINAL DATA
for i in test_develop_index:
    x_train_bound=np.delete(_xtrain_bound,i,0)
    x_train_unbound=np.delete(_xtrain_unbound,i,0)
    y_train_bound=np.delete(_ytrain_bound,i,0)
    y_train_unbound=np.delete(_ytrain_unbound,i,0)
#THOSE NOT DELETED CONSTITUTE TRAINING SET
x_train=np.concatenate((x_train_bound,x_train_unbound), axis=0)
y_train=np.concatenate((y_train_bound,y_train_unbound), axis=0)

#INDEXED ITEMS GO INTO TEST SET
x_test_bound=_xtrain_bound[test_index]
x_test_unbound=_xtrain_unbound[test_index]
x_test=np.concatenate((x_test_bound,x_test_unbound), axis=0)

y_test_bound=_ytrain_bound[test_index]
y_test_unbound=_ytrain_unbound[test_index]
y_test=np.concatenate((y_test_bound,y_test_unbound), axis=0)

#INDEXED ITEMS GO INTO DEVELOP SET
x_develop_bound=_xtrain_bound[develop_index]
x_develop_unbound=_xtrain_unbound[develop_index]
x_develop=np.concatenate((x_develop_bound,x_develop_unbound), axis=0)

y_develop_bound=_ytrain_bound[develop_index]
y_develop_unbound=_ytrain_unbound[develop_index]
y_develop=np.concatenate((y_develop_bound,y_develop_unbound), axis=0)

x_train = x_train.reshape(x_train.shape[0], 12, 1)
x_test = x_test.reshape(x_test.shape[0], 12, 1)
x_develop = x_develop.reshape(x_develop.shape[0], 12, 1)

# y_train = y_train.reshape(y_train.shape[0], 1)
# y_test = y_test.reshape(y_test.shape[0], 1)
# y_develop = y_develop.reshape(y_develop.shape[0], 1)

def count_bound(current_list):
    current_count = 0
    for i in range(len(current_list)):
        if(current_list[i] == 1):
            current_count += 1
    return current_count

print("train num bound (num 1):", count_bound(y_train))
print("train num unbound (num 0):", len(y_train) - count_bound(y_train))
print("develop num bound (num 1):", count_bound(y_develop))
print("develop num unbound (num 0):", len(y_develop) - count_bound(y_develop))
print("test num bound (num 1):", count_bound(y_test))
print("test num unbound (num 0):", len(y_test) - count_bound(y_test))

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_develop shape:", x_develop.shape)
print("y_develop shape:", y_develop.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)



#CONVOLUTIONAL NET
input_layer=tf.keras.layers.Input(shape=(12,1))
nn = tf.keras.layers.Conv1D(3, 3, activation='relu')(input_layer)
nn = tf.keras.layers.Dropout(.95)(nn)
# nn = tf.keras.layers.Conv1D(12, 1, activation='relu')(nn)
# nn=tf.keras.layers.MaxPooling1D(1)(nn)
# nn = tf.keras.layers.Conv1D(12, 1, activation='relu')(nn)
# nn = tf.keras.layers.Conv1D(12, 1, activation='relu')(nn)
nn=tf.keras.layers.GlobalAveragePooling1D()(nn)
nn = tf.keras.layers.Dense(25)(nn)
nn = tf.keras.layers.LeakyReLU()(nn)
nn = tf.keras.layers.Dropout(.25)(nn)
nn = tf.keras.layers.Dense(25)(nn)
nn = tf.keras.layers.LeakyReLU()(nn)
output_layer=tf.keras.layers.Dense(1, activation='sigmoid')(nn)
model=tf.keras.models.Model(input_layer,output_layer)
model.summary()

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=10,validation_data=(x_develop,y_develop)) #Have Keras make a test/validation split for us

# model_m = Sequential()
# model.add(Conv1D(1, kernel_size=5, input_shape = (12,)))
# model_m.add(Conv1D(100, 10, activation='relu'))
# model_m.add(MaxPooling1D(3))
# model_m.add(Conv1D(160, 10, activation='relu'))
# model_m.add(Conv1D(160, 10, activation='relu'))
# model_m.add(Dropout(0.5))
# model_m.add(Dense(num_classes, activation='sigmoid'))
# print(model_m.summary())


#DENSE NEURAL NET
# input_layer=tf.keras.layers.Input(shape=(12,))
# nn = tf.keras.layers.Dense(25)(input_layer)
# nn = tf.keras.layers.LeakyReLU()(nn)
# nn = tf.keras.layers.Dense(25)(nn)
# nn = tf.keras.layers.LeakyReLU()(nn)
# nn = tf.keras.layers.Dropout(.9)(nn)
# nn = tf.keras.layers.Dense(25)(nn)
# nn = tf.keras.layers.LeakyReLU()(nn)
# output_layer = tf.keras.layers.Dense(1,activation='sigmoid')(nn)

# model=tf.keras.models.Model(input_layer,output_layer)
# model.summary()
# model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
#
# model.fit(x_train,y_train,epochs=10,validation_data=(x_develop,y_develop)) #Have Keras make a test/validation split for us
#

# def plot_history(history):
#     plt.plot(history.history['loss'],label='Train')
#     plt.plot(history.history['val_loss'],label='Develop')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.ylim((0,1.5*np.max(history.history['val_loss'])))
#     plt.legend()
#     plt.show()
# plot_history(history)

INPUT_FILE .close()
