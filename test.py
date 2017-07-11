import h5py

train = h5py.File('data/generated_synthetic_data/train.h5','r')
valid = h5py.File('data/generated_synthetic_data/valid.h5','r')

(x_train,y_train) = (train[u'data'][:],train[u'label'][:])
(x_valid,y_valid) = (valid[u'data'][:],valid[u'label'][:])

# ploting a sample
#plt.imshow(x_train[0][0])
#plt.show()

# reshaping data
x_train = x_train.reshape(x_train.shape[0],x_train.shape[3],x_train.shape[2],x_train.shape[1])
x_valid = x_valid.reshape(x_valid.shape[0],x_valid.shape[3],x_valid.shape[2],x_valid.shape[1])
