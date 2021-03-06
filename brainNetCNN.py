from __future__ import print_function, division

import matplotlib.pyplot as plt
plt.interactive(False)
import tensorflow as tf
import h5py
from scipy.stats import pearsonr
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras import optimizers, callbacks, regularizers, initializers
from E2E_conv import *
from utils import *
from injury import ConnectomeInjury


batch_size = 14
dropout = 0.5
momentum = 0.9
lr = 0.01
decay = 0.0005
noise_weight = 1

reg = regularizers.l2(decay)
kernel_init = initializers.he_uniform()





# Loading synthetic data
injuryconnectome = ConnectomeInjury()
x_train,y_train = injuryconnectome.generate_injury(n_samples=1000,noise_weight=noise_weight)
x_valid,y_valid = injuryconnectome.generate_injury(n_samples=300,noise_weight=noise_weight)


# ploting a sample
#plt.imshow(x_train[0][0])
#plt.show()

# reshaping data
x_train = x_train.reshape(x_train.shape[0],x_train.shape[3],x_train.shape[2],x_train.shape[1])
x_valid = x_valid.reshape(x_valid.shape[0],x_valid.shape[3],x_valid.shape[2],x_valid.shape[1])

print(x_train)



# Model architecture

model = Sequential()
model.add(E2E_conv(2,32,(2,90),kernel_regularizer=reg,input_shape=(90,90,1),input_dtype='float32',data_format="channels_last"))
print("First layer output shape :"+str(model.output_shape))
model.add(LeakyReLU(alpha=0.33))
#print(model.output_shape)
model.add(E2E_conv(2,32,(2,90),kernel_regularizer=reg,data_format="channels_last"))
print(model.output_shape)
model.add(LeakyReLU(alpha=0.33))
model.add(Convolution2D(64,(1,90),kernel_regularizer=reg,data_format="channels_last"))
model.add(LeakyReLU(alpha=0.33))
model.add(Convolution2D(256,(90,1),kernel_regularizer=reg,data_format="channels_last"))
model.add(LeakyReLU(alpha=0.33))
#print(model.output_shape)
model.add(Dropout(0.5))
model.add(Dense(128,kernel_regularizer=reg,kernel_initializer=kernel_init))
#print(model.output_shape)
model.add(LeakyReLU(alpha=0.33))
#print(model.output_shape)
model.add(Dropout(0.5))
model.add(Dense(30,kernel_regularizer=reg,kernel_initializer=kernel_init))
model.add(LeakyReLU(alpha=0.33))
#print(model.output_shape)
model.add(Dropout(0.5))
model.add(Dense(2,kernel_regularizer=reg,kernel_initializer=kernel_init))
model.add(Flatten())
model.add(LeakyReLU(alpha=0.33))
model.summary()
#print(model.output_shape)


opt = optimizers.SGD(momentum=momentum,nesterov=True,lr=lr)
model.compile(optimizer=opt,loss='mean_squared_error',metrics=['mae'])
csv_logger = callbacks.CSVLogger('BrainCNN.log')
command = str(raw_input("Train or predict ? [t/p]"))
if command == "t":
    print("Training for noise = "+str(noise_weight))
    history=model.fit(x_train,y_train,nb_epoch=1000,verbose=1,callbacks=[csv_logger])
    model.save_weights("Weights/BrainCNNWeights_noise_"+str(noise_weight)+".h5")
else:
    print("[*] Predicting and printing results for the models trained :")
    get_results_from_models(model,noises = [0,0.0625,0.125,0.25])













