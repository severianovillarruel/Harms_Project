#!/usr/bin/env python3

from tensorflow import keras
from tensorflow.keras import layers
import json
import numpy as np

TEST_FILE = open("/home/severiano/harms_proj/files/test_predictions.txt","r")
JSON_FILE = open("/home/severiano/harms_proj/data/aaindex-pca.json", "r")

data = json.load(JSON_FILE)

aa_convert = {"A":0, "R":1, "N":2, "D":3, "C":4,
              "E":5, "Q":6, "G":7, "H":8, "I":9,
              "L":10, "K":11, "M":12, "F":13, "P":14,
              "S":15,"T":16, "W":17, "Y":18, "V":19}
pc1 = data["aaindex_pca_1"]["values"]
pc2 = data["aaindex_pca_2"]["values"]

hA6_model = keras.models.load_model('hA6.model')
hA6_model.summary()

pred_dict = {}
for line in TEST_FILE:
    line = line.strip().split()
    x_mer = line[1]
    twleve_mer_lst = []
    for i in range((len(x_mer)-12) + 1):
        twleve_mer_lst.append(x_mer[(0+i):(12+i)])
    pred_dict[line[0]] = twleve_mer_lst

for key in pred_dict:
    print(key)
    print(pred_dict[key])

for key in pred_dict:
    print(key)
    key_lst = []
    for twelve_mer in pred_dict[key]:
        data = []
        aa_dict1 = []
        aa_dict2 = []
        aa_dict3 = []
        for i in twelve_mer:
            if i in aa_convert:
                aa_dict1.append(aa_convert[i]/19)
            if i in pc1:
                aa_dict2.append(pc1[i])
            if i in pc2:
                aa_dict3.append(pc2[i])
        data.append(aa_dict1)
        data.append(aa_dict2)
        data.append(aa_dict3)
        key_lst.append(data)

    key_lst = np.array(key_lst)
    key_lst = key_lst.reshape(key_lst.shape[0], 3, 12, 1)
    print(hA6_model.predict(key_lst))

        # print(data)

    # key.append(data)
    # key = np.array(key)
    # key = key.reshape(key.shape[0], 3, 12, 1)

JSON_FILE.close()
TEST_FILE.close()
