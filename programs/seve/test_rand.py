#!/usr/bin/env python3

from tensorflow import keras
from tensorflow.keras import layers
import json
import numpy as np

RAND_FILE = open("/home/severiano/harms_proj/files/random_12mers.txt","r")
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

rand_test_lst = []
for line in RAND_FILE:
    line = line.strip()
    data = []
    aa_dict1 = []
    aa_dict2 = []
    aa_dict3 = []
    for i in line:
        if i in aa_convert:
            aa_dict1.append(aa_convert[i]/19)
        if i in pc1:
            aa_dict2.append(pc1[i])
        if i in pc2:
            aa_dict3.append(pc2[i])
    data.append(aa_dict1)
    data.append(aa_dict2)
    data.append(aa_dict3)
    rand_test_lst.append(data)

rand_test_lst = np.array(rand_test_lst)
rand_test_lst = rand_test_lst.reshape(rand_test_lst.shape[0], 3, 12, 1)
predict = hA6_model.predict(rand_test_lst)

count = 0
for i in predict:
    if(i < .2):
        count += 1

print(count/10000*100)

JSON_FILE.close()
RAND_FILE.close()
