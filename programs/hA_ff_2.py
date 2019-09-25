#!/usr/bin/env python3

import numpy as np
from sklearn import preprocessing
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
                                                      #ASSUMING POSITIVE HIT IS LESS THAN NEGATIVE HIT
NUM_BINARY_TEST_SET = math.ceil(positive_hit * .1)    #EX: 499, least possible amount of '1s' in test set
NUM_BINARY_TRAIN_SET = math.floor(positive_hit * .9)  #EX: 4485, option 1

INPUT_FILE_1.close()

####################################################################

INPUT_FILE_2 = open(RAW_FILE, "r")

#INITIALIZE EMPTY TENSORS
DATA_TEST_TENSOR = np.empty(((NUM_BINARY_TEST_SET*2),seq_len))      #(998, 12)    499 SEQs, 499 SEQs
LABEL_TEST_TENSOR = np.empty((NUM_BINARY_TEST_SET*2),)              #(998,)       499 1s, 499 0s
DATA_TRAIN_TENSOR = np.empty(((NUM_BINARY_TRAIN_SET*2), seq_len))   #(8970, 12)   4485 SEQs, 4485 SEQs
LABEL_TRAIN_TENSOR = np.empty((NUM_BINARY_TRAIN_SET*2),)            #(8970,)      4485 1s, 4485 0s

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
        if i in aa_convert.keys():
            aa_vector.append(aa_convert[i])

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
le = preprocessing.LabelEncoder()
DATA_TEST_TENSOR_LE = DATA_TEST_TENSOR.le.fit_transform
print('DATA_TEST_TENSOR_LE', DATA_TEST_TENSOR_LE)
print('DATA_TEST_TENSOR_LE.shape', DATA_TEST_TENSOR_LE.shape)
# print('DATA_TEST_TENSOR.shape', DATA_TEST_TENSOR.shape)
# print('DATA_TEST_TENSOR', DATA_TEST_TENSOR)
# print('LABEL_TEST_TENSOR.shape', LABEL_TEST_TENSOR.shape)
# print('LABEL_TEST_TENSOR', LABEL_TEST_TENSOR)
# print('DATA_TRAIN_TENSOR', DATA_TRAIN_TENSOR)
# print('LABEL_TRAIN_TENSOR', LABEL_TRAIN_TENSOR)

# print('num test count =', test_count)
# print('num train count =', train_count)
# print('num positive hits =', positive_hit) #4984 = 499 + 4485
# print('num negative hits =', negative_hit)
# print('num lines =', line_counter)
# print('NUM_BINARY_TEST_SET =', NUM_BINARY_TEST_SET) #499 = .1 * 4984
# print('NUM_BINARY_TRAIN_SET =', NUM_BINARY_TRAIN_SET) #4485 = .9 * 4984
