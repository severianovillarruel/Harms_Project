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

#LOAD MODELS
hA6_model = keras.models.load_model('hA6.model')
hA6_model.summary()
hA5_model = keras.models.load_model('hA5.model')
hA5_model.summary()
aA5A6_model = keras.models.load_model('aA5A6.model')
aA5A6_model.summary()
alt_model = keras.models.load_model('alt.model')
alt_model.summary()

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

ha6_predict = hA6_model.predict(rand_test_lst)
ha5_predict = hA5_model.predict(rand_test_lst)
a5a6_predict = aA5A6_model.predict(rand_test_lst)
alt_predict = alt_model.predict(rand_test_lst)

count_5 = 0
for i in ha5_predict:
    if(i < .2):
        count_5 += 1

print("for the hA5 model ",count_5/1000*100, "% bound")

count_6 = 0
for i in ha6_predict:
    if(i < .2):
        count_6 += 1

print("for the hA6 model ",count_6/1000*100, "% bound")

count_56 = 0
for i in a5a6_predict:
    if(i < .2):
        count_56 += 1

print("for the A5A6 model ",count_56/1000*100, "% bound")

count_alt = 0
for i in alt_predict:
    if(i < .2):
        count_alt += 1

print("for the alt model",count_alt/1000*100, "% bound")

JSON_FILE.close()
RAND_FILE.close()
