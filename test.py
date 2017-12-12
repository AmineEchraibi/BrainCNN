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
from injury import ConnectomeInjury
import numpy as np



batch_size = 14
dropout = 0.5
momentum = 0.9
lr = 0.01
decay = 0.0005
noise_weight = 0.0625

reg = regularizers.l2(decay)
kernel_init = initializers.he_uniform()






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

def get_symmetric_noise(m, n):
    """Return a random noise image of size m x n with values between 0 and 1."""

    # Generate random noise image.
    noise_img = np.random.rand(m, n)

    # Make the noise image symmetric.
    noise_img = noise_img + noise_img.T

    # Normalize between 0 and 1.
    noise_img = (noise_img - noise_img.min()) / (noise_img.max() - noise_img.min())

    assert noise_img.max() == 1  # Make sure is between 0 and 1.
    assert noise_img.min() == 0
    assert (noise_img.T == noise_img).all()  # Make sure symmetric.

    return noise_img


def simulate_injury(X, weight_A, sig_A, weight_B, sig_B):
    denom = (np.ones(X.shape) + (weight_A * sig_A)) * (np.ones(X.shape) + (weight_B * sig_B))
    X_sig_AB = np.divide(X, denom)
    return X_sig_AB


def apply_injury_and_noise(X, Sig_A, weight_A, Sig_B, weight_B, noise_weight):
    """Returns a symmetric, signed, noisy, adjacency matrix with simulated injury from two sources."""

    X_sig_AB = simulate_injury(X, weight_A, Sig_A, weight_B, Sig_B)

    # Get the noise image.
    noise_img = get_symmetric_noise(X.shape[0], X.shape[1])

    # Weight the noise image.
    weighted_noise_img = noise_img * noise_weight

    # Add the noise to the original image.
    X_sig_AB_noise = X_sig_AB + weighted_noise_img

    assert (X_sig_AB_noise.T == X_sig_AB_noise).all()  # Make sure still is symmetric.

    return X_sig_AB_noise


def generate_injury_signatures(X_mn, n_injuries, r_state):
    """Generates the signatures that represent the underlying signal in our synthetic experiments.

    d : (integer) the size of the input matrix (assumes is size dxd)
    """

    # Get the strongest regions, which we will apply simulated injuries
    sig_indexes = [2, 50]
    d = X_mn.shape[0]

    S = []

    # Create a signature for
    for idx, sig_idx in enumerate(sig_indexes):
        # Okay, let's make some signature noise vectors.
        A_vec = r_state.rand((d))
        # B_vec = np.random.random((n))

        # Create the signature matrix.
        A = np.zeros((d, d))
        A[:, sig_idx] = A_vec
        A[sig_idx, :] = A_vec
        S.append(A)

        assert (A.T == A).all()  # Check if matrix is symmetric.

    return np.asarray(S)


def sample_injury_strengths(n_samples, X_mn, A, B, noise_weight):
    """Returns n_samples connectomes with simulated injury from two sources."""
    mult_factor = 10

    n_classes = 2

    # Range of values to predict.
    n_start = 0.5
    n_end = 1.4
    # amt_increase = 0.1

    # These will be our Y.
    A_weights = np.random.uniform(n_start, n_end, [n_samples])
    B_weights = np.random.uniform(n_start, n_end, [n_samples])

    X_h5 = np.zeros((n_samples, 1, X_mn.shape[0], X_mn.shape[1]), dtype=np.float32)
    Y_h5 = np.zeros((n_samples, n_classes), dtype=np.float32)

    for idx in range(n_samples):
        w_A = A_weights[idx]
        w_B = B_weights[idx]

        # Get the matrix.
        X_sig = apply_injury_and_noise(X_mn, A, w_A * mult_factor, B, w_B * mult_factor, noise_weight)

        # Normalize.
        X_sig = (X_sig - X_sig.min()) / (X_sig.max() - X_sig.min())

        # Put in h5 format.
        X_h5[idx, 0, :, :] = X_sig
        Y_h5[idx, :] = [w_A, w_B]

    return X_h5, Y_h5


def load_base_connectome():
    X_mn = scipy.io.loadmat("data/base.mat")
    X_mn = X_mn['X_mn']
    return X_mn
def get_symmetric_noise(m, n):
    """Return a random noise image of size m x n with values between 0 and 1."""

    # Generate random noise image.
    noise_img = np.random.rand(m, n)

    # Make the noise image symmetric.
    noise_img = noise_img + noise_img.T

    # Normalize between 0 and 1.
    noise_img = (noise_img - noise_img.min()) / (noise_img.max() - noise_img.min())

    assert noise_img.max() == 1  # Make sure is between 0 and 1.
    assert noise_img.min() == 0
    assert (noise_img.T == noise_img).all()  # Make sure symmetric.

    return noise_img

def simulate_injury(X, weight_A, sig_A, weight_B, sig_B):
    denom = (np.ones(X.shape) + (weight_A * sig_A)) * (np.ones(X.shape) + (weight_B * sig_B))
    X_sig_AB = np.divide(X, denom)
    return X_sig_AB

def apply_injury_and_noise(X, Sig_A, weight_A, Sig_B, weight_B, noise_weight):
    """Returns a symmetric, signed, noisy, adjacency matrix with simulated injury from two sources."""

    X_sig_AB = simulate_injury(X, weight_A, Sig_A, weight_B, Sig_B)

    # Get the noise image.
    noise_img = get_symmetric_noise(X.shape[0], X.shape[1])

    # Weight the noise image.
    weighted_noise_img = noise_img * noise_weight

    # Add the noise to the original image.
    X_sig_AB_noise = X_sig_AB + weighted_noise_img

    assert (X_sig_AB_noise.T == X_sig_AB_noise).all()  # Make sure still is symmetric.

    return X_sig_AB_noise


def generate_injury_signatures(X_mn, n_injuries, r_state):
        """Generates the signatures that represent the underlying signal in our synthetic experiments.

        d : (integer) the size of the input matrix (assumes is size dxd)
        """

        # Get the strongest regions, which we will apply simulated injuries
        sig_indexes = [2,50]
        d = X_mn.shape[0]

        S = []

        # Create a signature for
        for idx, sig_idx in enumerate(sig_indexes):
            # Okay, let's make some signature noise vectors.
            A_vec = r_state.rand((d))
            # B_vec = np.random.random((n))

            # Create the signature matrix.
            A = np.zeros((d, d))
            A[:, sig_idx] = A_vec
            A[sig_idx, :] = A_vec
            S.append(A)

            assert (A.T == A).all()  # Check if matrix is symmetric.

        return np.asarray(S)

def sample_injury_strengths(n_samples, X_mn, A, B, noise_weight):
        """Returns n_samples connectomes with simulated injury from two sources."""
        mult_factor = 10

        n_classes = 2

        # Range of values to predict.
        n_start = 0.5
        n_end = 1.4
        # amt_increase = 0.1

        # These will be our Y.
        A_weights = np.random.uniform(n_start, n_end, [n_samples])
        B_weights = np.random.uniform(n_start, n_end, [n_samples])

        X_h5 = np.zeros((n_samples, 1, X_mn.shape[0], X_mn.shape[1]), dtype=np.float32)
        Y_h5 = np.zeros((n_samples, n_classes), dtype=np.float32)

        for idx in range(n_samples):
            w_A = A_weights[idx]
            w_B = B_weights[idx]

            # Get the matrix.
            X_sig = apply_injury_and_noise(X_mn, A, w_A * mult_factor, B, w_B * mult_factor, noise_weight)

            # Normalize.
            X_sig = (X_sig - X_sig.min()) / (X_sig.max() - X_sig.min())

            # Put in h5 format.
            X_h5[idx, 0, :, :] = X_sig
            Y_h5[idx, :] = [w_A, w_B]

        return X_h5, Y_h5
import numpy as np
import scipy
r_state = np.random.RandomState(41)
X_mn = load_base_connectome()
S = generate_injury_signatures(X_mn=X_mn,n_injuries=2,r_state=r_state)
X,Y = sample_injury_strengths(1000,X_mn,S[0],S[1],noise_weight)
print(X.shape)
print(Y.shape)

def load_base_connectome():
    X_mn = scipy.io.loadmat("data/base.mat")
    X_mn = X_mn['X_mn']
    return X_mn

X = X.reshape(X.shape[0],X.shape[3],X.shape[2],X.shape[1])
model.fit(X,Y,nb_epoch=1000,verbose=1)
model.save_weights("Weights/BrainCNNWeights_Visualization.h5")