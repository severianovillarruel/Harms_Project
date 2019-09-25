#!/usr/bin/env python3

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

line_counter = 0
positive_hit = 0
negative_hit = 0
for line in INPUT_FILE_1:
    line = line.strip("\n")
    line = line.split("\t")

    label = line[1]                            #EX: -2.1296834226e+00 = 1, 5.7343690848e-01 = 0
    label = float(label)

    if line_counter == 0:
         seq_len = len(line[0])

    if label <= -2:
        label = 1
        positive_hit += 1
    else:
        label = 0
        negative_hit += 1
    line_counter += 1
                                                      #ASSUMING POSITIVE HIT IS LESS THAN NEGATIVE HIT
NUM_BINARY_TEST_SET = math.ceil(positive_hit * .1)    #EX: 499, least possible amount of '1s' in test set
NUM_BINARY_TRAIN_SET = math.floor(positive_hit * .9)  #EX: 4485, option 1

INPUT_FILE_1.close()

####################################################################

INPUT_FILE_2 = open(RAW_FILE, "r")

#INITIALIZE EMPTY TENSORS
DATA_TEST_TENSOR = []               #(998, 12)    499 SEQs, 499 SEQs
for i in range(NUM_BINARY_TEST_SET*2):
    DATA_TEST_TENSOR.append(0)
LABEL_TEST_TENSOR = []              #(998,)       499 1s, 499 0s
for i in range(NUM_BINARY_TEST_SET*2):
    LABEL_TEST_TENSOR.append(0)
DATA_TRAIN_TENSOR = []              #(8970, 12)   4485 SEQs, 4485 SEQs
for i in range(NUM_BINARY_TRAIN_SET*2):
    DATA_TRAIN_TENSOR.append(0)
LABEL_TRAIN_TENSOR = []             #(8970,)      4485 1s, 4485 0s
for i in range(NUM_BINARY_TRAIN_SET*2):
    LABEL_TRAIN_TENSOR.append(0)

####################################################################################

#DIVIDE DATA AND LABELS TRANING AND TEST TENSORS
ones_count = 1
zeros_count = 1
test_count = 0
train_count = 0
for line in INPUT_FILE_2:
    line = line.strip("\n")
    line = line.split("\t")

    #MAKE DATA
    aa_vector = []
    seq = line[0]                               #EX: GSGTAGLHRIVF = [0.35, 0.75, 0.35, 0.8, 0.0, 0.35, 0.5, 0.4, 0.05, 0.45, 0.95, 0.65]
    for i in seq:
        aa_vector.append(i)

    #MAKE LABELS
    label = line[1]                            #EX: -2.1296834226e+00 = 1, 5.7343690848e-01 = 0
    label = float(label)
    if label <= -2:
        label = 1
    else:
        label = 0

    #DISTRIBUTE ZEROS BETWEEN TEST AND TRAIN TENSORS
    if label == 0:
        if zeros_count <= NUM_BINARY_TEST_SET:      #if zeros_count <= 499:
            DATA_TEST_TENSOR[test_count] = aa_vector
            LABEL_TEST_TENSOR[test_count] = label
            zeros_count += 1
            test_count += 1
        if NUM_BINARY_TEST_SET < zeros_count <= NUM_BINARY_TEST_SET + NUM_BINARY_TRAIN_SET:  #if 499 < zeros_count <= 4984:
            DATA_TRAIN_TENSOR[train_count] = aa_vector
            LABEL_TRAIN_TENSOR[train_count] = label
            zeros_count += 1
            train_count += 1
            if zeros_count == NUM_BINARY_TRAIN_SET*2:  #if zeros_count == 8970:
                pass

    #DISTRIBUTE ONES BETWEEN TEST AND TRAIN TENSORS
    if label == 1:
        if ones_count <= NUM_BINARY_TEST_SET:  #if ones_count <= 499:
            DATA_TEST_TENSOR[test_count] = aa_vector
            LABEL_TEST_TENSOR[test_count] = label
            ones_count += 1
            test_count += 1
        if NUM_BINARY_TEST_SET < ones_count <= NUM_BINARY_TEST_SET + NUM_BINARY_TRAIN_SET:   #if 499 < ones_count <= 4984:
            DATA_TRAIN_TENSOR[train_count] = aa_vector
            LABEL_TRAIN_TENSOR[train_count] = label
            ones_count += 1
            train_count += 1
            if ones_count == NUM_BINARY_TRAIN_SET*2:   #if ones_count == 8970:
                pass


print('DATA_TEST_TENSOR', DATA_TEST_TENSOR)
print('LABEL_TEST_TENSOR', LABEL_TEST_TENSOR)
print('DATA_TRAIN_TENSOR', DATA_TRAIN_TENSOR)
print('LABEL_TRAIN_TENSOR', LABEL_TRAIN_TENSOR)
print(DATA_TEST_TENSOR[0])
print(LABEL_TEST_TENSOR[0])
#
#
# ##################################################################################
#
# #ONE HOT ENCODE DATA AND LABELS
# enc = OneHotEncoder(handle_unknown='ignore')
#
# enc.fit(DATA_TEST_TENSOR)
# enc.fit(DATA_TRAIN_TENSOR)
#
# enc.categories_
# #DATA ONE HOT
# x_test = enc.transform(DATA_TEST_TENSOR).toarray()
# x_train = enc.transform(DATA_TRAIN_TENSOR).toarray()
# #LABEL ONE HOT
# y_test = np.asarray(LABEL_TEST_TENSOR).astype('float32')  #y_train
# y_train = np.asarray(LABEL_TRAIN_TENSOR).astype('float32') #y_test
#
# np.random.shuffle(x_test)
# np.random.shuffle(x_train)
# np.random.shuffle(y_test)
# np.random.shuffle(y_train)
# # print('x_train', x_train)
# # print('y_train', y_train)
# # print('x_test', x_test)
# # print('y_test', y_test)
#
# # print(x_train[0])
# # print(y_train[0])
# # print('1', x_train.shape)
# # print('2', y_train.shape)
# # print('3', x_test.shape)
# # print('4', y_test.shape)
#
# ##################################################################################
#
# #MAKE LAYERS (NETWORK ARCHITECTURE)
#
# #OPTION 1
# # model = Sequential()
# # model.add(Dense(32, activation='relu', input_dim=12))
# # model.add(Dense(1, activation='sigmoid'))
# # model.compile(optimizer='rmsprop',
# #               loss='binary_crossentropy',
# #               metrics=['accuracy'])
# #OPTION 2
# #model = Sequential()
# #model.add(Dense(16, activation='tanh', input_dim=12)) #ndim = 2
# #model.add(Dense(16, activation='tanh'))
# #model.add(Dense(1, activation='sigmoid')) #no activation if I am doing regression ()
# #
# #model.compile(loss='categorical_crossentropy',
# #              optimizer='sgd',
# #              metrics=['accuracy'])
#
# #TRAIN THE MODEL
# # model.fit(x_train, y_train, epochs=5)
#
# #EVALUATE THE MODEL
# # model.evaluate(x_test, y_test)
#
# #####################################################################
# #BUILD LAYERS
# model = models.Sequential()
# model.add(layers.Dense(16, activation='relu', input_shape=(240,))) #SIZE OF SECOND DIMENSION AFTR ONE HOT
# model.add(layers.Dense(16, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))
#
# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
#
# ######################################################################
# #FIT TO MODEL
# x_val = x_train[:4485] #HALF OF NUM_BINARY_TRAIN_SET*2
# partial_x_train = x_train[4485:]
# y_val = y_train[:4485]
# partial_y_train = y_train[4485:]
#
# history = model.fit(partial_x_train, partial_y_train, epochs=10, batch_size=512, validation_data=(x_val, y_val), shuffle=True)
# #history = model.fit(x_train, y_train, epochs=10, batch_size=512, validation_data=.1, shuffle=True)
#
# #model.fit(x_train, y_train, epochs=5)
#
# ######################################################################
#
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
# plt.show()
#
# print(model.predict(x_test))
#
INPUT_FILE_2.close()
