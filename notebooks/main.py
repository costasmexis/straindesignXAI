import warnings
warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)

import sys
sys.path.append("../src")

# Add autoreload magic command

import pickle
import numpy as np
import pandas as pd
from IPython.display import display

# from pycaret.regression import *
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from model_selection import modelSelection
warnings.simplefilter(action="ignore", category=FutureWarning)


file_path = "../data/EDD_isoprenol_production.csv"

INPUT_VARS = ["ACCOAC", "MDH", "PTAr", "CS", "ACACT1r", "PPC", "PPCK", "PFL"]
RESPONSE_VARS = ["Value"]

df = pd.read_csv(file_path, index_col=0)
df = df[INPUT_VARS + RESPONSE_VARS]
df[INPUT_VARS] = df[INPUT_VARS].astype(int)
X_train = df[INPUT_VARS]
y_train = df[RESPONSE_VARS].values.ravel()
print(f"Shape of the data: {df.shape}")

# Set df as X_train
X_train = df.copy()

# normalize the data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_train = pd.DataFrame(X_train, columns=df.columns)
X_train.shape


import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.keras.layers import Input, Dense, Lambda, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers.legacy import Adam

def get_error_term(v1, v2, _rmse=True):
    if _rmse:
        return np.sqrt(np.mean((v1 - v2) ** 2, axis=1))
    #return MAE
    return np.mean(abs(v1 - v2), axis=1)

def sample(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


original_dim = X_train.shape[1]
input_shape = (original_dim,)
intermediate_dim = 10
latent_dim = 3

# encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = Dense(64, activation='relu')(inputs)
x = Dense(16, activation='relu')(x)
x = Dense(16, activation='relu')(x)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)
# use the reparameterization trick and get the output from the sample() function
z = Lambda(sample, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
encoder = Model(inputs, z, name='encoder')

# decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(16, activation='relu')(latent_inputs)
x = Dense(16, activation='relu')(x)
x = Dense(64, activation='relu')(x)
outputs = Dense(original_dim, activation='sigmoid')(x)
# Instantiate the decoder model:
decoder = Model(latent_inputs, outputs, name='decoder')

# full VAE model
outputs = decoder(encoder(inputs))
vae_model = Model(inputs, outputs, name='vae_mlp')

# the KL loss function:
def vae_loss(x, x_decoded_mean):
    # compute the average MSE error, then scale it up, ie. simply sum on all axes
    reconstruction_loss = K.sum(K.square(x - x_decoded_mean))
    # compute the KL loss
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.square(K.exp(z_log_var)), axis=-1)
    # return the average loss over all 
    total_loss = K.mean(reconstruction_loss + kl_loss)    
    #total_loss = reconstruction_loss + kl_loss
    return total_loss

print(f'Shape of the data: {X_train.shape}')
opt = Adam(learning_rate=0.0001)
vae_model.compile(optimizer=opt, loss=vae_loss)
results = vae_model.fit(X_train, X_train, validation_split=0.2, shuffle=True, epochs=5000, batch_size=32, verbose=1)

# Get the latent representation of the input data
X_train_encoded = encoder.predict(X_train)

X_train_encoded = pd.DataFrame(X_train_encoded, index=df.index, columns=[f"z_{i}" for i in range(latent_dim)])
X_train_encoded['prod'] = y_train