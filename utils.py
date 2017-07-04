from __future__ import print_function, division

import matplotlib.pyplot as plt
import numpy as np
import keras
import keras.backend as K
from keras.models import  Model
from keras.layers import Conv1D,Conv2D,MaxPooling2D,MaxPool1D,Dropout,Dense,Input,Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.pooling import AveragePooling1D
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras import initializers
from keras import regularizers
from scipy.io import loadmat

# feature maps {"name":dimension}

# parameters

batch_size = 14
dropout = 0.5
momentum = 0.9
lr = 0.01
decay = 0.0005
# Generating phantom data using injury.py
input = loadmat('data/base.mat')['X_mn']


reg = regularizers.l2(decay)
kernel_init = initializers.he_uniform()

# utility functions
def E2E_conv(input,num_output,kernel_h,kernel_w):
    conv_dx1 = Conv1D(num_output,strides=1,kernel_size=(1,kernel_w),kernel_regularizer=reg,kernel_initializer=kernel_init)(input)
    conv_1xd = Conv1D(num_output,strides=1,kernel_size=(kernel_h,1),kernel_initializer=kernel_init,kernel_regularizer=reg)(input)
    d=np.size(conv_1xd, 1)
    conv_1xd_to_dxd = np.ones((d,d))
    conv_dx1_to_dxd = np.ones((d,d))
    for i in range(d):
        conv_1xd_to_dxd[i:] = conv_1xd
        conv_dx1_to_dxd[:i] = conv_dx1
    return conv_1xd_to_dxd + conv_dx1_to_dxd

def E2N_conv(input,num_output,kernel_h,kernel_w):
    conv_1xd = Conv1D(num_output,kernel_size=(1,kernel_w),kernel_regularizer=reg,kernel_initializer=kernel_init,strides=1)(input)
    return conv_1xd

def FC_layer(input,num_output):
    FC = Dense(num_output,kernel_initializer=kernel_init,kernel_regularizer=reg)
    return FC

def G2N_conv(input):
    pass



# BrainNetCNN


def main():
    x = input
    x = E2E_conv(x,32,1,90)
    X = LeakyReLU(0.33)(X)
    x = E2E_conv(x,64,32,90)
    X = LeakyReLU(0.3)(X)
    x = E2N_conv(x,256,1,1)
    X = LeakyReLU(0.3)(X)
    x = Dropout(0.5)(x)
    x = FC_layer(x,128)
    X = LeakyReLU(0.3)(X)
    x = Dropout(0.5)(x)
    x = FC_layer(x,30)
    X = LeakyReLU(0.3)(X)
    preds = FC_layer(x,2)
    X = LeakyReLU(0.3)(X)
    # Set the model
    model = Model(inputs=input, outputs=[preds])


    # set the optimizer and compile model
    opt = optimizers.SGD(momentum=momentum,decay=decay,nesterov=True,lr=lr)
    model.compile(optimizer=opt,loss='mean_squared_error',metrics=['mae'])
    model.summary()


    # Training
    model.fit(x=X_train,y=y_train,batch_size=32,epochs=25,verbose=1,shuffle=True,validation_split=0.2)