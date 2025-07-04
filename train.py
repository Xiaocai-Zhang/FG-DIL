# =======================
# Imports and Environment
# =======================
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Dense, Dropout, Flatten, Conv1D, Activation, BatchNormalization,
    GlobalAveragePooling1D, Reshape, Multiply, GaussianDropout
)
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K
import config

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Set random seeds for reproducibility
seed = 989
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)


# ====================
# Define Blocks
# ====================
def cwab_block(x, reduction=16):
    channels = x.shape[-1]
    squeeze = GlobalAveragePooling1D()(x)
    excitation = Dense(channels // reduction, activation='relu')(squeeze)
    excitation = Dense(channels, activation='sigmoid')(excitation)
    excitation = Reshape((1, channels))(excitation)
    return Multiply()([x, excitation])


class FGNetworks:
    def dccb_block(self, x, dilation_rate, nb_filters, kernel_size, dropout, padding):
        init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
        assert padding in ['causal', 'same']

        conv1 = Conv1D(filters=nb_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding=padding,
                       kernel_initializer=init, kernel_regularizer=tf.keras.regularizers.L2(1e-4))
        batch1 = BatchNormalization(axis=-1)
        ac1 = Activation('relu')
        drop1 = GaussianDropout(dropout)

        conv2 = Conv1D(filters=nb_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding=padding,
                       kernel_initializer=init, kernel_regularizer=tf.keras.regularizers.L2(1e-4))
        batch2 = BatchNormalization(axis=-1)
        ac2 = Activation('relu')
        drop2 = GaussianDropout(dropout)

        downsample = Conv1D(filters=nb_filters, kernel_size=1, padding='same', kernel_initializer=init)
        ac3 = Activation('relu')

        pre_x = x

        x = conv1(x)
        x = batch1(x)
        x = ac1(x)
        x = drop1(x)

        x = conv2(x)
        x = batch2(x)
        x = ac2(x)
        x = drop2(x)

        x = cwab_block(x)

        if pre_x.shape[-1] != x.shape[-1]:  # to match the dimensions
            pre_x = downsample(pre_x)

        out = ac3(pre_x + x)
        return out

    def Network(self, x, num_channels, kernel_size, dropout):
        for i, filters in enumerate(num_channels):
            dilation_rate = 2 ** i
            x = self.dccb_block(x, dilation_rate, filters, kernel_size, dropout, padding='causal')
        return x


# ===============
# Loss & Scaling
# ===============
def adaptive_loss(y_true, y_pred, alpha=1.5):
    error = K.abs(y_true - y_pred)
    return K.mean(K.pow(error, alpha) / alpha, axis=-1)


def normalize(y):
    y_min, y_max = np.min(y), np.max(y)
    y_scaled = 2 * (y - y_min) / (y_max - y_min) - 1
    return y_scaled, y_min, y_max


def denormalize(y_scaled, y_min, y_max):
    return ((y_scaled + 1) / 2) * (y_max - y_min) + y_min


# ===================
# Load and Preprocess
# ===================
x = np.load("./ETPB/inp_all_array_t05.npy")
y = np.load("./ETPB/oup_all_array_t05.npy")
y = y.flatten()
y, y_min, y_max = normalize(y)

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=seed)

# ===============
# Model Building
# ===============
input = Input(shape=x.shape[1:])
x = Conv1D(config.h_dim, kernel_size=1, activation='relu')(input)
fg_net = FGNetworks()
x = fg_net.Network(x, config.num_channels, config.kernel_tcn, config.dropout_rate)
x = Flatten()(x)
output = Dense(config.n_action)(x)

model = Model(inputs=input, outputs=output)
model.compile(
    optimizer=RMSprop(learning_rate=config.learning_rate),
    loss=lambda y_true, y_pred: adaptive_loss(y_true, y_pred, alpha=1.5)
)

# ==========
# Training
# ==========
checkpoint = ModelCheckpoint(
    filepath=config.modelSavePath,
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    save_weights_only=True
)

history = model.fit(
    x_train, y_train,
    batch_size=config.batch_size,
    epochs=config.epochs,
    validation_data=(x_val, y_val),
    verbose=2,
    callbacks=[checkpoint]
)

# ==========
# Evaluation
# ==========
model.load_weights(config.modelSavePath)
y_pred_norm = model.predict(x_val, verbose=0)
y_pred_norm = np.clip(y_pred_norm, -1, 1)

y_pred_denormalized = (y_pred_norm + 1) / 2 * (y_max - y_min) + y_min

y_pred = denormalize(y_pred_norm, y_min, y_max)
y_val = (y_val+1) / 2 * (y_max - y_min) + y_min

mae_original = np.mean(np.abs(y_pred_denormalized.flatten() - y_val.flatten()))
rmse_original = np.sqrt(np.mean((y_pred_denormalized.flatten() - y_val.flatten()) ** 2))
mape_original = np.mean(np.abs((y_pred_denormalized.flatten() - y_val.flatten()) / y_val.flatten()))

print("Mean Absolute Error (MAE):", mae_original)
print("Root Mean Square Error (RMSE):", rmse_original)
print("Mean Absolute Percentage Error (MAPE):", mape_original)

