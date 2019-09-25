#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

####################################################
#GENERATE DATA
X=np.random.uniform(-5,15,size=(10000,5))
def func(X):
    return np.sin(X[:,0]) #Ignore all other input have the output only depend on the first dimention
Y=func(X)

print(X)
print(Y)
#
# ####################################################
# #MAKE LAYERS
# input_layer=tf.keras.layers.Input(shape=(5,))
#
# hidden_layer = tf.keras.layers.Dense(20)(input_layer)
# activation_layer = tf.keras.layers.LeakyReLU()(hidden_layer)
#
# hidden_layer = tf.keras.layers.Dense(20)(activation_layer)
# activation_layer = tf.keras.layers.LeakyReLU()(hidden_layer)
#
# hidden_layer = tf.keras.layers.Dense(20)(activation_layer)
# activation_layer = tf.keras.layers.LeakyReLU()(hidden_layer)
#
# hidden_layer = tf.keras.layers.Dense(20)(activation_layer)
# activation_layer = tf.keras.layers.LeakyReLU()(hidden_layer)
#
# hidden_layer = tf.keras.layers.Dense(20)(activation_layer)
# activation_layer = tf.keras.layers.LeakyReLU()(hidden_layer)
#
# output_layer = tf.keras.layers.Dense(1)(activation_layer)
#
# sine_model=tf.keras.models.Model(input_layer,output_layer)
#
# ####################################################
# #COMPILE AND FIT
# sine_model.compile(loss='mse',optimizer='adam')
# sine_model.fit(X,Y,epochs=50,validation_split=0.5) #Have Keras make a test/validation split for us
#
# ####################################################
# #TEST
# #Create some Random 5-d data
# X_test=np.random.uniform(0,10,size=(100,5)) #data dim (in this case 5) is used to make test
# #Set the first dimention to be a line
# X_test[:,0]=np.linspace(-5,15,100)
#
# #Get the True distribution from our test function
# Y_test=func(X_test)
# Y_pred=sine_model.predict(X_test)
#
# ####################################################
# #PLOT
# plt.scatter(X_test[:,0],Y_pred,label='prediction')
# plt.scatter(X_test[:,0],Y_test,label='truth')
# plt.xlabel('X[:,0]')
# plt.ylabel('Y')
# plt.legend()
# plt.savefig('plot.png')
