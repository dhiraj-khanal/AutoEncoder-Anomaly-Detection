from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, ZeroPadding2D, Activation

model = Sequential()

# Convolutional layer 1 with 128 feature maps
model.add(Conv2D(128, kernel_size=(4, 4), padding='same', input_shape=(37, 37, 4)))
model.add(Activation('relu'))

# Convolutional layer 2 with 64 feature maps
model.add(Conv2D(64, kernel_size=(4, 4), padding='same'))
model.add(Activation('relu'))

# Max-pooling layer with 2x2 reduction factor
model.add(MaxPooling2D(pool_size=(2, 2)))

# Convolutional layer 3 with 64 feature maps
model.add(Conv2D(64, kernel_size=(4, 4), padding='same'))
model.add(Activation('relu'))

# Convolutional layer 4 with 64 feature maps
model.add(Conv2D(64, kernel_size=(4, 4), padding='same'))
model.add(Activation('relu'))

# Max-pooling layer with 2x2 reduction factor
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the output of the last pooling layer
model.add(Flatten())

# Three fully connected dense layers of 64, 256 and 256 neurons each
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Activation('relu'))

# Output layer of 2 neurons with softmax activation function
model.add(Dense(2, activation='softmax'))
model.summary()