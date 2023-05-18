import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

def img(folder_name):
    X0 = []
    for i in os.listdir(folder_name):
        full_path = os.path.join(folder_name, i)
        #print(full_path)
        file_name = os.path.splitext(os.path.basename(i))[0]
        #print(file_name)
        if file_name == '.DS_Store':
            continue
        img = image.load_img(full_path, target_size=(16, 16))
        x = image.img_to_array(img)
        x = tf.convert_to_tensor(x)
        x = tf.cast(x, tf.float32) / 255.0
        x = tf.image.resize(x, [16,16])
        X0.append(x)
    X0 = tf.convert_to_tensor(X0)
    return X0

# Create generators for training, validation and testing
# Image resizing is done by the generator so a folder with any sized-images can be used
# The named directory must contain one or more subfolders, path should look like lightjet_train/class1/img1.jpg...

#batch_size = 20

train = np.load("/Users/hanczvs/Desktop/ae/lightjet_train/train.npy", allow_pickle=True)
print(np.shape(train))
test = np.load("/Users/hanczvs/Desktop/ae/lightjet_test/test.npy")
print(np.shape(test))

n_train = np.shape(train)[0]
n_test = np.shape(test)[0]

train = tf.convert_to_tensor(train)
test = tf.convert_to_tensor(test)

train = tf.reshape(train, [n_train, 16, 16, 1])
test = tf.reshape(test, [n_test, 16, 16, 1])

train = tf.cast(train, tf.float32) / 255.0
test = tf.cast(test, tf.float32) / 255.0

print(tf.shape(train))
print(tf.shape(test))

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

ae_model.build(input_shape = (69, 16, 16, 1))
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
    scale = [1, 1]
    mse_bce_loss = scale[0]*mse_loss(input, output) + scale[1]*bce_loss(input, output)
    return mse_bce_loss

# mse_loss = tf.keras.losses.MeanSquaredError()
# bce_loss = tf.keras.losses.BinaryCrossentropy()

ae_model.compile(
    optimizer   = tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss        = mse_bce_loss,
    metrics     = [
        tf.keras.metrics.MeanSquaredError(),
        tf.keras.metrics.BinaryCrossentropy()
    ]
)
#save_best = keras.callbacks.ModelCheckpoint(model_filepath, monitor='val_loss', save_best_only=True, mode='min')

ae_model.fit(
    train, train,
    epochs     = 20,
    batch_size = 20,
    validation_data = (test, test),
    shuffle = True,
    #callbacks  = [save_best]
);
#ae_model.save(model_filepath)
model = ae_model

# Test the model by viewing a sample of original and reconstructed images
data_list = []
batch_index = 0
while batch_index <= train_generator.batch_index:
    data = train_generator.next()
    data_list.append(data[0])
    batch_index = batch_index + 1

predicted = model.predict(data_list[0])
no_of_samples = 4
_, axs = plt.subplots(no_of_samples, 2, figsize=(5, 8))
axs = axs.flatten()
imgs = []
for i in range(no_of_samples):
    imgs.append(data_list[i][i])
    imgs.append(predicted[i])
for img, ax in zip(imgs, axs):
    ax.imshow(img)
plt.savefig('recon.png')
print(f"Error on validation set:{model.evaluate_generator(validation_generator)}, error on anomaly set:{model.evaluate_generator(anomaly_generator)}")
