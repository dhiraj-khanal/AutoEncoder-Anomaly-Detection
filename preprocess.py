import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
'''
# Load the matrices from the .npy files
matrix1 = np.load('210000pts.npy')
matrix2 = np.load('220000pts.npy')
matrix3 = np.load('230000pts.npy')
matrix4 = np.load('240000pts.npy')
matrix5 = np.load('250000pts.npy')
train = np.load('light_train_prev.npy')

# Concatenate the matrices along the first dimension
combined_matrix = np.concatenate((matrix1, matrix2, matrix3, matrix4, matrix5, train), axis=0)

# Save the combined matrix to a .npy file
np.save('light_train.npy', combined_matrix)
'''

train = np.load('light_train.npy')

def average_plot(train, test):
    # Calculate average matrices
    avg_matrix_QCD = np.mean(train + 0.000001, axis=0)
    avg_matrix_W = np.mean(test + 0.000001, axis=0)

    # Create a 1x2 subplot to put the plots side by side
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    # Create a color plot with log normalization for avg_matrix_QCD
    im1 = axes[0].imshow(avg_matrix_QCD, cmap='viridis', norm=LogNorm(vmin=avg_matrix_QCD.min(), vmax=avg_matrix_QCD.max()), extent=[-1, 1, -1, 1])
    axes[0].set_title('QCD Jet')
    axes[0].set_xlabel(r'$\eta$')
    axes[0].set_ylabel(r'$\phi$')

    # Create a color plot with log normalization for avg_matrix_W
    im2 = axes[1].imshow(avg_matrix_W, cmap='viridis', norm=LogNorm(vmin=avg_matrix_W.min(), vmax=avg_matrix_W.max()), extent=[-1, 1, -1, 1])
    axes[1].set_title('W+ Jet')
    axes[1].set_xlabel(r'$\eta$')
    axes[1].set_ylabel(r'$\phi$')

    # Add a colorbar to the right of the plots
    cbar_ax = fig.add_axes([0.92, 0.15, 0.03, 0.7])
    fig.colorbar(im2, cax=cbar_ax)
    cbar_ax.set_ylabel(r'log($p_T$)')

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.3, right=0.9)

    # Save the plot
    plt.savefig("average_plot.png")
    plt.show()





#train = np.load('train_light.npy')
test = np.load('wjet.npy')

average_plot(train, test)
#average_plot(test, "wjet")

#print(np.max(test))