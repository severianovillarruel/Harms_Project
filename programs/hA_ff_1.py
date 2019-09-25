#!/usr/bin/env python3

import numpy as np
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

RAW_FILE = "../hA_files/hA5.rand.tsv"
INPUT_FILE_1 = open(RAW_FILE, "r")

aa_convert = {"A":0, "R":1, "N":2, "D":3, "C":4,
              "E":5, "Q":6, "G":7, "H":8, "I":9,
              "L":10, "K":11, "M":12, "F":13, "P":14,
              "S":15,"T":16, "W":17, "Y":18, "V":19}


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

NUM_BINARY_TEST_SET = math.ceil(positive_hit * .1)    #EX: 499, least possible amount of '1s' in test set
NUM_BINARY_TRAIN_SET = math.floor(positive_hit * .9)  #EX: 4485, option 1

INPUT_FILE_1.close()

print(NUM_BINARY_TEST_SET)
print(NUM_BINARY_TRAIN_SET)
print(seq_len)
# print('There are ' , int(positive_hit), ' positive hits (1s)')
# print('There are ' , int(negative_hit), ' negative hits (0s)')
# print('num test samples = ' , int(THEORETICAL_NUM_TEST_SAMPLES))
# print('num training samples = ' , int(THEORETICAL_NUM_TRAIN_SAMPLES))
# ###########################################################

INPUT_FILE_2 = open(RAW_FILE, "r")

#INITIALIZE EMPTY TENSORS FOR label TENSORS
DATA_TEST_TENSOR = np.empty(((NUM_BINARY_TEST_SET*2),seq_len))      #(998, 12)    499 1s, 499 0s
LABEL_TEST_TENSOR = np.empty((NUM_BINARY_TEST_SET*2),)              #(998,)       499 1s, 499 0s
DATA_TRAIN_TENSOR = np.empty(((NUM_BINARY_TRAIN_SET*2), seq_len))   #(8970, 12)   4485 1s, 4485 0s
LABEL_TRAIN_TENSOR = np.empty((NUM_BINARY_TRAIN_SET*2),)            #(8970,)      4485 1s, 4485 0s

# print('DATA_TRAIN_TENSOR', DATA_TRAIN_TENSOR.shape)
# print('LABEL_TRAIN_TENSOR', LABEL_TRAIN_TENSOR.shape)
# print('DATA_TEST_TENSOR', DATA_TEST_TENSOR.shape)
# print('LABEL_TEST_TENSOR', LABEL_TEST_TENSOR.shape)
#MAKE SEQ AND LABELS INTO TENSORS
counter = 0
ones_count = 1
zeros_count = 1
test_count = 0
train_count = 0
for line in INPUT_FILE_2:
    line = line.strip("\n")
    line = line.split("\t")

    #SEQ_TESNOR
    aa_vector = []
    seq = line[0]                               #EX: GSGTAGLHRIVF = [0.35, 0.75, 0.35, 0.8, 0.0, 0.35, 0.5, 0.4, 0.05, 0.45, 0.95, 0.65]
    for i in seq:
        if i in aa_convert.keys():
            aa_vector.append(aa_convert[i])

    #LABEL_TENSOR
    label = line[1]                            #EX: -2.1296834226e+00 = 1, 5.7343690848e-01 = 0
    label = float(label)
    if label <= -2:
        label = 1
    else:
        label = 0

    #print(seq, aa_vector, str(label))
    #print(counter)

    #PUT 10% OF 1s IN TEST SET
    if ones_count <= NUM_BINARY_TEST_SET: #499/998
        if label == 1:
            DATA_TEST_TENSOR[test_count] = aa_vector
            LABEL_TEST_TENSOR[test_count] = label
            ones_count += 1
            test_count += 1
            #print('test ones', str(test_count))

    if zeros_count <= NUM_BINARY_TEST_SET: #499/998
        if label == 0:
            DATA_TEST_TENSOR[test_count] = aa_vector
            LABEL_TEST_TENSOR[test_count] = label
            zeros_count += 1
            test_count += 1
            #print('test zeros', str(test_count))

    if 499 < ones_count <= 4485: #4485/8970
        if label == 1:
            DATA_TRAIN_TENSOR[train_count] = aa_vector
            LABEL_TRAIN_TENSOR[train_count] = label
            ones_count += 1
            train_count += 1
        print('train ones', str(train_count))

    if 499 < zeros_count <= 4485: #4485/8970
        if label == 0:
            DATA_TRAIN_TENSOR[train_count] = aa_vector
            LABEL_TRAIN_TENSOR[train_count] = label
            zeros_count += 1
            train_count += 1
        print('train zeros', str(train_count))

    counter += 1
print(counter)
# print('DATA_TEST_TENSOR', DATA_TEST_TENSOR)
# print('LABEL_TEST_TENSOR', LABEL_TEST_TENSOR)
# print('DATA_TRAIN_TENSOR', DATA_TRAIN_TENSOR)
# print('LABEL_TRAIN_TENSOR', LABEL_TRAIN_TENSOR)
