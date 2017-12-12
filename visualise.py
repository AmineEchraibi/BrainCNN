from __future__ import print_function, division

import matplotlib.pyplot as plt
plt.interactive(False)
from scipy.stats import pearsonr
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras import optimizers, callbacks, regularizers, initializers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from E2E_conv import *

import numpy as np
import matplotlib.pylab as plt
from vis.visualization import visualize_activation
from nilearn import plotting

from sklearn.cross_validation import StratifiedKFold


def plot_matrices(matrices, matrix_kind):
    n_matrices = len(matrices)
    plt.figure(figsize=(n_matrices * 4, 4))
    for n_subject, matrix in enumerate(matrices):
        plt.subplot(1, n_matrices, n_subject + 1)
        matrix = matrix.copy()  # avoid side effects
        # Set diagonal to zero, for better visualization
        np.fill_diagonal(matrix, 0)
        vmax = np.max(np.abs(matrix))
        plt.imshow(matrix, vmin=-vmax, vmax=vmax, cmap='RdBu_r',
                   interpolation='nearest')
        plt.title('{0}, subject {1}'.format(matrix_kind, n_subject))
from nilearn import datasets


adhd_data = datasets.fetch_adhd(n_subjects=20)
msdl_data = datasets.fetch_atlas_msdl()
msdl_coords = msdl_data.region_coords
n_regions = len(msdl_coords)
print('MSDL has {0} ROIs, part of the following networks :\n{1}.'.format(
    n_regions, msdl_data.networks))
from nilearn import input_data

masker = input_data.NiftiMapsMasker(
    msdl_data.maps, resampling_target="data", t_r=2.5, detrend=True,
    low_pass=.1, high_pass=.01, memory='nilearn_cache', memory_level=1)
adhd_subjects = []
pooled_subjects = []
site_names = []
adhd_labels = []  # 1 if ADHD, 0 if control
for func_file, confound_file, phenotypic in zip(
        adhd_data.func, adhd_data.confounds, adhd_data.phenotypic):
    time_series = masker.fit_transform(func_file, confounds=confound_file)
    pooled_subjects.append(time_series)
    is_adhd = phenotypic['adhd']
    if is_adhd:
        adhd_subjects.append(time_series)

    site_names.append(phenotypic['site'])
    adhd_labels.append(is_adhd)

print('Data has {0} ADHD subjects.'.format(len(adhd_subjects)))
from nilearn.connectome import ConnectivityMeasure



conn_measure = ConnectivityMeasure(kind="tangent")
x_train = conn_measure.fit_transform(pooled_subjects)
print(x_train.shape)
print(len(adhd_labels))
y_train = np.array(adhd_labels,dtype="float32")
print(y_train.shape)

# Prediction ###############################


batch_size = 14
dropout = 0.5
momentum = 0.9
lr = 0.001
decay = 0.0005

reg = regularizers.l2(decay)
kernel_init = initializers.he_uniform()



# Model architecture

model = Sequential()
model.add(E2E_conv(2,8,(2,39),kernel_regularizer=reg,input_shape=(39,39,1),input_dtype='float32',data_format="channels_last"))
print("First layer output shape :"+str(model.output_shape))
model.add(LeakyReLU(alpha=0.33))
#print(model.output_shape)
print(model.output_shape)
model.add(LeakyReLU(alpha=0.33))
model.add(Convolution2D(32,(1,39),kernel_regularizer=reg,data_format="channels_last"))
model.add(LeakyReLU(alpha=0.33))
model.add(Convolution2D(90,(39,1),kernel_regularizer=reg,data_format="channels_last"))
model.add(LeakyReLU(alpha=0.33))
#print(model.output_shape)
model.add(Dropout(0.5))
model.add(Dense(64,kernel_regularizer=reg,kernel_initializer=kernel_init))
#print(model.output_shape)
model.add(LeakyReLU(alpha=0.33))
#print(model.output_shape)
model.add(Dropout(0.5))
model.add(Dense(10,kernel_regularizer=reg,kernel_initializer=kernel_init))
model.add(LeakyReLU(alpha=0.33))
#print(model.output_shape)
model.add(Dropout(0.5))
model.add(Dense(1,kernel_regularizer=reg,kernel_initializer=kernel_init))
model.add(Flatten())
model.add(Activation('softmax'))
model.summary()
#print(model.output_shape)


opt = optimizers.SGD(nesterov=True,lr=lr)
model.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy'])
csv_logger = callbacks.CSVLogger('predict_age.log')

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
x_train,x_test,y_train,y_test = train_test_split(x_train,y_train,test_size=0.33,random_state=42)

command = str(raw_input("Train or predict ? [t/p]"))
if command == "t":
    print("Training the model ...")
    history=model.fit(x_train,y_train,batch_size=1,nb_epoch=1000,verbose=1,callbacks=[csv_logger])
    model.save_weights("Weights/BrainCNNWeights_categ.h5")
else:
    print("[*] Predicting and printing results for the models trained :")
    model.load_weights("Weights/BrainCNNWeights_categ.h5")
    heatmap = visualize_activation(model, layer_idx=-1, filter_indices=0, seed_input=x_test[0])
    print(heatmap.shape)
    plt.interactive(False)
    plotting.plot_connectome()








