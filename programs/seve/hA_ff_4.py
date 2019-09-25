#!/usr/bin/env python3

#IMPORTS AND LOAD DATASET
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import math
from keras import models
from keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

RAW_FILE = "../files/hA5.rand.tsv"
INPUT_FILE_1 = open(RAW_FILE, "r")

####################################################################

#GET COUNT OF TARGET HITS
NUM_DATAPOINTS = 0
NUM_BOUND = 0
NUM_UNBOUND = 0
for line in INPUT_FILE_1:
    line = line.strip("\n")
    line = line.split("\t")

    label = line[1]                            #EX: -2.1296834226e+00 = 1, 5.7343690848e-01 = 0
    label = float(label)

    if NUM_DATAPOINTS == 0:
         seq_len = len(line[0])

    if label <= -2:
        label = 1
        NUM_BOUND += 1
    else:
        label = 0
        NUM_BOUND += 1
    NUM_DATAPOINTS += 1

#.9/.1 IS 90% TRAINING/10% TEST
TEN_PERCENT_OF_BOUND = math.ceil(NUM_BOUND * .1)    #EX: 499. Least possible amount of hits ('1s') in test set.
NINETY_PERCENT_OF_BOUND = math.floor(NUM_BOUND * .9)  #EX: 4485. .9/.1 set of misses (0s) (option 1). (option 2 is to keep all of the misses)

#for binary classification adjust data to make sure number of successes = number of failures.

INPUT_FILE_1.close()

# print("Number of Data Point: " , str(datapoints_counter))
# print("Number of Peptides that Bound: " , str(num_bound))
# print("Number of Peptides that Did Not Bind: ", str(num_unbound))

####################################################################

NUM_TEST_DATAPOINTS = TEN_PERCENT_OF_BOUND + TEN_PERCENT_OF_BOUND
NUM_TRAIN_DATAPOINTS = NINETY_PERCENT_OF_BOUND + NINETY_PERCENT_OF_BOUND

####################################################################
#OPEN FILE AND INITIALIZE EMPTY TENSORS
INPUT_FILE_2 = open(RAW_FILE, "r")

x_test = []               #(998, 12)    499 SEQs, 499 SEQs
for i in range(NUM_TEST_DATAPOINTS):
    x_test.append(0)
y_test = []              #(998,)       499 1s, 499 0s
for i in range(NUM_TEST_DATAPOINTS):
    y_test.append(0)
x_train = []              #(8970, 12)   4485 SEQs, 4485 SEQs
for i in range(NUM_TRAIN_DATAPOINTS):
    x_train.append(0)
y_train = []             #(8970,)      4485 1s, 4485 0s
for i in range(NUM_TRAIN_DATAPOINTS):
    y_train.append(0)

####################################################################################
aa_convert = {"A":0, "R":1, "N":2, "D":3, "C":4,
              "E":5, "Q":6, "G":7, "H":8, "I":9,
              "L":10, "K":11, "M":12, "F":13, "P":14,
              "S":15,"T":16, "W":17, "Y":18, "V":19}

#DIVIDE DATA AND LABELS TRANING AND TEST TENSORS
ones_count = 1
zeros_count = 1
test_count = 0
train_count = 0
for line in INPUT_FILE_2:
    line = line.strip("\n")
    line = line.split("\t")

    #MAKE DATA
    data = []                              #EX: GSGTAGLHRIVF
    for i in line[0]:
        if i in aa_convert.keys():
            data.append(aa_convert[i]/20)
    #MAKE LABELS
    label = line[1]                            #EX: -2.1296834226e+00 = 1, 5.7343690848e-01 = 0
    label = float(label)
    if label <= -2:
        label = 1
    else:
        label = 0

    #DISTRIBUTE ZEROS BETWEEN TEST AND TRAIN TENSORS
    if label == 0:
        if zeros_count <= NUM_TEST_DATAPOINTS/2:      #if zeros_count <= 499:
            x_test[test_count] = data
            y_test[test_count] = label
            zeros_count += 1
            test_count += 1
        if NUM_TEST_DATAPOINTS/2 < zeros_count <= NUM_TEST_DATAPOINTS/2 + NUM_TRAIN_DATAPOINTS/2:  #if 499 < zeros_count <= 4984:
            x_train[train_count] = data
            y_train[train_count] = label
            zeros_count += 1
            train_count += 1
            if zeros_count == NUM_TRAIN_DATAPOINTS:  #if zeros_count == 8970:
                pass

    #DISTRIBUTE ONES BETWEEN TEST AND TRAIN TENSORS
    if label == 1:
        if ones_count <= NUM_TEST_DATAPOINTS/2:  #if ones_count <= 499:
            x_test[test_count] = data
            y_test[test_count] = label
            ones_count += 1
            test_count += 1
        if NUM_TEST_DATAPOINTS/2 < ones_count <= NUM_TEST_DATAPOINTS/2 + NUM_TRAIN_DATAPOINTS/2:   #if 499 < ones_count <= 4984:
            x_train[train_count] = data
            y_train[train_count] = label
            ones_count += 1
            train_count += 1
            if ones_count == NUM_TRAIN_DATAPOINTS:   #if ones_count == 8970:
                pass

print('DATA_TEST_TENSOR', x_test)
print('LABEL_TEST_TENSOR', y_test)
print('DATA_TRAIN_TENSOR', x_train)
print('LABEL_TRAIN_TENSOR', y_train)
print(x_train[0])
print(y_test[0])
# #####################################################################
# #BUILD LAYERS
# model = models.Sequential()
# model.add(layers.Dense(32, activation='relu', input_shape=(240,))) #240 IS SIZE OF SECOND DIMENSION AFTR ONE HOT
# model.add(layers.Dense(32, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))
#
# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
#
# ######################################################################
# #FIT TO MODEL
#
# x_val = x_train[:4485] #4485 (HALF OF NUM_BINARY_TRAIN_SET*2)
# partial_x_train = x_train[4485:]
# y_val = y_train[:4485]
# partial_y_train = y_train[4485:]
#
# history = model.fit(partial_x_train, partial_y_train, epochs=100, batch_size=250, validation_data=(x_val, y_val))
# ######################################################################
# #PLOT LOSS
# history_dict = history.history
# acc = history.history['acc']
# loss_values = history_dict['loss']
# val_loss_values = history_dict['val_loss']
# val_acc = history.history['val_acc']
# epochs = range(1, len(acc) + 1)
# plt.plot(epochs, loss_values, '.', label='Training loss')
# plt.plot(epochs, val_loss_values, '.', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.savefig('plot.png')
#
# ######################################################################
# #USE PREDICTIONS TO TEST THE MODEL
# print(model.predict(x_test))
#
# results = model.evaluate(x_test, y_test)
# print(results)

INPUT_FILE_2.close()
