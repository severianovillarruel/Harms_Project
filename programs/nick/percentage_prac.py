# Nick Wagner
# practice

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from random import random
from sys import version


files_dir='/Users/nick/Desktop/BGMP/machine_learning/Harms_Project/files'


yes = []
no = []


with open(files_dir + '/hA6.tsv', 'r') as readFile:
    for line in readFile:
        line = line.strip().split()
        line[1] = float(line[1])
        
        if(line[1] <= -2):
            yes.append(line)
        else:
            no.append(line)

print(len(yes))
print(len(no))

# train_index=[]
# develop_index=[]
# for i in range(len(_xtrain)):
#     if random() <0.65:
#         train_index.append(i)
#     else:
#         develop_index.append(i)