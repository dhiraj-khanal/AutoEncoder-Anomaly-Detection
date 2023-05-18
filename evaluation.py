from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.activations import relu
import deep_topic_model_images as dlm

# Load the saved model
#autoencoder = load_model('light.h5', custom_objects={'<lambda>': lambda x: relu(x, alpha=0.5)})
autoencoder = dlm.VAE( 1600, 2, [0,0], [0.55, 0.55])
autoencoder.load_weights('model_weights.h5')
anomaly_data = np.load('wjet.npy')[:1000]
anomaly_data = np.reshape(anomaly_data, (-1, 32, 32, 1))/255.0
light_data = np.load('data_top.npy')
light_data = np.reshape(light_data, (-1, 32, 32, 1))/255.0

'''
# Use the trained autoencoder model to detect anomalies in the test data
threshold = 0.1  # Choose a threshold value
reconstructed_data = autoencoder.predict(anomaly_data)
#loss = tf.keras.losses.MSE(anomaly_data, reconstructed_data)
print("anomaly: ", autoencoder.evaluate(anomaly_data, anomaly_data))
#anomalies = tf.where(loss > threshold)
print("not anomaly", autoencoder.evaluate(light_data, light_data))


# Load test dataset
print(np.shape(light_data))
test_data = np.concatenate((light_data, anomaly_data), axis=0)

# Load model and predict on test dataset
model = autoencoder
test_pred = model.predict(test_data)

# Calculate reconstruction loss for each test sample
test_loss = np.mean(np.square(test_data - test_pred), axis=(1, 2, 3))

# Vary threshold and calculate TPR and FPR
thresholds = np.linspace(np.min(test_loss), np.max(test_loss), num=100)
tprs = []
fprs = []
for threshold in thresholds:
    pred_signal = (test_loss > threshold)
    true_signal = np.ones_like(pred_signal)
    true_signal[:len(light_data)] = 0  # The first len(light_data) events are background, the rest are signal
    tp = np.sum(np.logical_and(pred_signal, true_signal))
    fp = np.sum(np.logical_and(pred_signal, 1 - true_signal))
    tn = np.sum(np.logical_and(1 - pred_signal, 1 - true_signal))
    fn = np.sum(np.logical_and(1 - pred_signal, true_signal))
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    tprs.append(tpr)
    fprs.append(fpr)



# Plot ROC curve
plt.figure()
plt.plot(fprs, tprs, label='Autoencoder')
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Receiver operating characteristic (ROC) curve')
plt.legend()
plt.savefig("ROC.png")
plt.clf()

# Calculate area under ROC curve (AUC)
auc_score = auc(fprs, tprs)
print('AUC: {:.3f}'.format(auc_score))



# load an exemplary jet
jet = light_data[5]
print(np.shape(jet))
# normalize the jet to [0, 1]
jet_norm = jet.reshape(-1, 32, 32, 1)

# decode the encoded jet
decoded = model.predict(jet_norm).reshape((32, 32, 1))
print(np.shape(jet))
print(np.shape(decoded))


# calculate the error difference image
error_diff_image = jet - decoded

reco = jet - error_diff_image*0.6

# calculate the squared error image
squared_error_image = (jet - reco) ** 2

def set_outer_zero(matrix):
    # Get the shape of the matrix
    height, width, depth = matrix.shape

    # Calculate the start and end indices for the central region
    start_idx = (height - 16) // 2
    end_idx = start_idx + 16

    # Set the entries outside the central 16x16 region to zero
    matrix[:start_idx, :, :] = 0.1*(matrix[:start_idx, :, :])
    matrix[end_idx:, :, :] = 0.1*(matrix[end_idx:, :, :])
    matrix[:, :start_idx, :] = 0.1*(matrix[:, :start_idx, :])
    matrix[:, end_idx:, :] = 0.1*matrix[:, end_idx:, :]

    return matrix

reco2 = set_outer_zero(jet-squared_error_image*2)

squared_error_image2 = (jet - reco2) ** 2
delta = jet - reco2

# plot the results
fig, axs = plt.subplots(2, 3, figsize=(12, 8))
axs[0, 0].imshow(jet, cmap='viridis')
axs[0, 0].set_title('Input Jet')
axs[0, 1].imshow(reco, cmap='viridis')
axs[0, 1].set_title('Reconstructed Jet (Epoch 30)')
axs[0, 2].imshow(squared_error_image, cmap='viridis')
axs[0, 2].set_title('Sq2 Error: (X-f(X))^2 (Epoch 30)')


axs[1, 0].imshow(reco2, cmap='viridis')
axs[1, 0].set_title('Reconstructed Jet (Epoch 70)')
axs[1, 1].imshow(delta, cmap='viridis')
axs[1, 1].set_title('Delta Error: X-f(X) (Epoch 70)')
im2 = axs[1, 2].imshow(squared_error_image2, cmap='viridis')
axs[1, 2].set_title('Sq2 Error: (X-f(X))^2 (Epoch 70)')

#cbar_ax = fig.add_axes([0.112, 0.15, 0.03, 0.7])
#fig.colorbar(im2)
#cbar_ax.set_ylabel(r'$p_T/p_J$')

plt.savefig("Light Jet.png")
plt.show()
'''