from __future__ import print_function, division

import matplotlib.pyplot as plt
plt.interactive(False)
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Convolution2D,MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.regularizers import l2
from keras import optimizers, callbacks

from keras.datasets import mnist

# loading data
(x_train,y_train),(x_test,y_test) = mnist.load_data()
reg = l2()


print (x_train.shape)

#plt.imshow(x_train[0])
#plt.show()

# reshaping and normalization
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /=255

print(x_train.shape)

# class reshaping
y_train = np_utils.to_categorical(y_train,10)
y_test = np_utils.to_categorical(y_test,10)
print(y_train)

# model architecture

model = Sequential()

model.add(Convolution2D(32,kernel_size=(3,3),strides=(1,1),input_shape=(28,28,1),activation='relu'))
print(model.output_shape)
model.add(Convolution2D(32,kernel_size=(3,3),strides=(1,1),activation='relu',kernel_regularizer=reg))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))
opt = optimizers.SGD(momentum=0.9,decay=0.0005,nesterov=True,lr=0.01)
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
csv_logger = callbacks.CSVLogger('training.log')
history=model.fit(x_train,y_train,batch_size=14,nb_epoch=25,verbose=1,callbacks=[csv_logger])
model.save('mnist.hdf5')
model.evaluate(x_test,y_test)


