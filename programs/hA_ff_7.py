#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import math
from keras import models
from keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from random import random

INPUT_FILE_1 = open("../files/hA5.rand.tsv", "r")

aa_convert = {"A":0, "R":1, "N":2, "D":3, "C":4,
              "E":5, "Q":6, "G":7, "H":8, "I":9,
              "L":10, "K":11, "M":12, "F":13, "P":14,
              "S":15,"T":16, "W":17, "Y":18, "V":19}

bound_array = []
unbound_array = []
data_point = []

NUM_DATAPOINTS = 0
NUM_BOUND_PEPS = 0
NUM_UNBOUND_PEPS = 0
for line in INPUT_FILE_1:
    line = line.strip("\n")
    line = line.split("\t")
    label = line[1]
    label = float(label)
    data = []
    for i in line[0]:
        if i in aa_convert.keys():
            data.append(aa_convert[i]/19)
    if label <= -2:
        label = 1
        NUM_BOUND_PEPS += 1
    else:
        label = 0
        NUM_UNBOUND_PEPS += 1

    NUM_DATAPOINTS += 1
print("Number of total datapoints: ", NUM_DATAPOINTS)
print("Number of bound peptides: ", NUM_BOUND_PEPS)
print("Number of unbound peptides: ", NUM_UNBOUND_PEPS)
print(data_point)

INPUT_FILE_1.close()
