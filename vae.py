
import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Conv2D, Conv2DTranspose

#batch_size = 20
train = np.load("/Users/hanczvs/Desktop/ae/lightjet_train/train.npy")
#print(train[0])
#print(np.shape(train))
val = np.load("/Users/hanczvs/Desktop/ae/lightjet_test/val.npy")
#print(np.shape(test))
n_train = np.shape(train)[0]
n_val = np.shape(val)[0]
print(n_train)
print(n_val)
train = tf.convert_to_tensor(train)
val = tf.convert_to_tensor(val)

train = tf.reshape(train, [n_train, 16, 16, 1])
val = tf.reshape(val, [n_val, 16, 16, 1])

train = tf.cast(train, tf.float32) / 255.0
val = tf.cast(val, tf.float32) / 255.0

#print(tf.shape(train))
#print(tf.shape(test))

class Autoencoder(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        ## TODO: Implement call function
        out = self.encoder(inputs)
        ret = self.decoder(out)
        return ret

## Some common keyword arguments you way want to use. HINT: func(**kwargs)
conv_kwargs = {
    "padding"             : "SAME",
    "activation"          : tf.keras.layers.LeakyReLU(alpha=0.2),
    "kernel_initializer"  : tf.random_normal_initializer(stddev=.1)
}


ae_model = Autoencoder(
    encoder = tf.keras.Sequential([tf.keras.layers.Conv2D(16, 3, strides=(2, 2), **conv_kwargs),
                                   tf.keras.layers.Conv2D(8, 3, strides=(2, 2), **conv_kwargs),
                                   tf.keras.layers.Conv2D(8, 3, strides=(2, 2), **conv_kwargs) \
 \
                                   ], name='ae_encoder'),
    decoder = tf.keras.Sequential([

        #tf.keras.layers.Dense(units = 1),
        tf.keras.layers.Conv2DTranspose(16, 3, strides=(2, 2), **conv_kwargs),
        tf.keras.layers.Conv2DTranspose(8, 3, strides=(2, 2), **conv_kwargs),
        tf.keras.layers.Conv2DTranspose(1, 3, strides=(2, 2), **conv_kwargs)
    ], name='ae_decoder')
    , name='autoencoder')

ae_model.build(input_shape = (n_train, 16, 16, 1))
initial_weights = ae_model.get_weights()

ae_model.summary()
ae_model.encoder.summary()
ae_model.decoder.summary()

model_filepath = '/Users/hanczvs/Desktop/ae/model1.h5'
ae_model.set_weights(initial_weights)  ## Resets model weights.

## Implement mse_bce_loss and figure out an appropriate balance scale
mse_loss = tf.keras.losses.MeanSquaredError()
bce_loss = tf.keras.losses.BinaryCrossentropy()

def mse_bce_loss(input, output):
    mse_bce_loss = mse_loss(input, output)
    return mse_bce_loss

# mse_loss = tf.keras.losses.MeanSquaredError()
# bce_loss = tf.keras.losses.BinaryCrossentropy()

ae_model.compile(
    optimizer   = tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss        = mse_bce_loss
)
#save_best = keras.callbacks.ModelCheckpoint(model_filepath, monitor='val_loss', save_best_only=True, mode='min')

ae_model.fit(
    train, train,
    epochs     = 20,
    batch_size = 20,
    validation_data = (val, val),
    shuffle = True,
    #callbacks  = [save_best]
);
#ae_model.save(model_filepath)
model = ae_model
anomaly = np.load("/Users/hanczvs/Desktop/ae/wjet/anomaly.npy")
anomaly = tf.convert_to_tensor(anomaly)
n_anomaly = np.shape(anomaly)[0]
anomaly = tf.reshape(anomaly, [n_anomaly, 16, 16, 1])
anomaly = tf.cast(anomaly, tf.float32) / 255.0

print(f"Error on validation set:{mse_bce_loss(val, model.evaluate(val))}, error on anomaly set:{mse_bce_loss(anomaly, model.evaluate(anomaly))}")