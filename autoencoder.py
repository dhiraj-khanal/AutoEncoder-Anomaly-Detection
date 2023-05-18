from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense, Reshape, Conv2DTranspose, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.activations import relu
from keras.callbacks import EarlyStopping
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Define the input shape

input_shape = (32, 32, 1)
file_name = "light_train.npy"

# Define the alpha parameter for the parameterized ReLU activation function
alpha_init = Random Variable
# Encoder layers
input_layer = Input(shape=input_shape)
x = Conv2D(10, kernel_size=(4, 4), padding='same', activation=lambda x: relu(x, alpha=alpha_init))(input_layer)
x = Conv2D(5, kernel_size=(4, 4), padding='same', activation=lambda x: relu(x, alpha=alpha_init))(x)
x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
#drop out layers
x = Conv2D(5, kernel_size=(4, 4), padding='same', activation=lambda x: relu(x, alpha=alpha_init))(x)
x = Conv2D(5, kernel_size=(4, 4), padding='same', activation=lambda x: relu(x, alpha=alpha_init))(x)
x = Flatten()(x)
x = Dense(10, activation=lambda x: relu(x, alpha=alpha_init))(x)
encoded = Dense(32)(x)
#bottle neck
# Decoder layers
x = Dense(100, activation=lambda x: relu(x, alpha=alpha_init))(encoded)
x = Dense(64, activation=lambda x: relu(x, alpha=alpha_init))(x)
x = Reshape((8, 8, 1))(x)
#drop out layers
x = Conv2D(5, kernel_size=(4, 4), padding='same', activation=lambda x: relu(x, alpha=alpha_init))(x)
x = Conv2D(5, kernel_size=(4, 4), padding='same', activation=lambda x: relu(x, alpha=alpha_init))(x)
x = UpSampling2D(size=(4, 4))(x)
x = Conv2DTranspose(1, kernel_size=(4, 4), padding='same')(x)


# Define the autoencoder model
autoencoder = Model(input_layer, x)


# Compile the model
autoencoder.compile(optimizer='adam', loss='mse')
# define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=20)

autoencoder.summary()
# Load the original .npy file
data = np.load(file_name)

# Reshape the data to include a single channel dimension
data = np.reshape(data, (-1, 32, 32, 1))
print(len(data))
# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2)

# Normalize the pixel values to be between 0 and 1
train_data = train_data.astype('float32') / 255.0
test_data = test_data.astype('float32') / 255.0


# Print the shapes of the data and labels for confirmation
print('Training data shape:', train_data.shape)
print('Testing data shape:', test_data.shape)


autoencoder.fit(train_data, train_data, epochs=250, batch_size=500, validation_data=(test_data, test_data), callbacks=[early_stopping])
test_loss = autoencoder.evaluate(test_data, test_data)
print(test_loss)

autoencoder.save('model.h5')

train_loss = autoencoder.history.history['loss']
val_loss = autoencoder.history.history['val_loss']

# Plot the training and validation losses over epochs
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.savefig('Training and Validation Losses')
plt.show()

