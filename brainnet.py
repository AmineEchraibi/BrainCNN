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

from keras.datasets import cifar10

#Paramaters

nb_classes = 10
(X_train,y_train),(X_test,y_test) = cifar10.load_data()
X_train =  X_train.astype(np.float32)
X_test  = X_test.astype(np.float32)
X_train /= 255
X_test /= 255
y_train = keras.utils.np_utils.to_categorical(y_train,nb_classes)
y_test = keras.utils.np_utils.to_categorical(y_train,nb_classes)
mean_x = X_train.mean(axis=0).mean(axis=0).mean(axis=0)
X_train -= mean_x

# Kernel initialization

kernel_init = initializers.he_uniform()

inputs = Input((32,32,3))
padding = "same"

reg = regularizers.l2(0.0005)

def conv_bn_relu(x,nb_filters,kernel_size=(3,3),padding = "same"):
    X = Conv2D(nb_filters,kernel_size=kernel_size,padding=padding,kernel_initializer=kernel_init,kernel_regularizer=reg)(x)
    X = BatchNormalization()(X)
    X = LeakyReLU(0.3)(X)
    return X

x = inputs
x = conv_bn_relu(x,nb_filters=64, kernel_size=(3,3))
x = MaxPooling2D((2,2),padding=padding,strides=(2,2))(x)

x = conv_bn_relu(x,nb_filters=64,kernel_size=(3,3))
x = MaxPooling2D((2,2),padding=padding,strides=(2,2))(x)


x = conv_bn_relu(x,nb_filters=64,kernel_size=(3,3))
x = MaxPooling2D((2,2),padding=padding,strides=(2,2))(x)


x = Flatten()(x)
X = Dropout(0.5)(x)

preds1 = Dense(nb_classes,activation="softmax",kernel_initializer=kernel_init,kernel_regularizer=reg)(x)

# Set the model
model = Model(inputs=inputs, outputs=[preds1])

# set the optimizer and compile model
opt = optimizers.SGD(momentum=0.9,decay=1e-6,nesterov=True,lr=0.01)
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

# Train the model

model.fit(x=X_train,y=y_train,batch_size=32,epochs=25,verbose=1,shuffle=True,validation_split=0.2)
model.save('test_model.h5')

# Predict

pred_classes = model.predict(x=X_test,batch_size=100,verbose=2)

# Accuracy


print(np.mean(pred_classes == y_test))