import h5py
import hdf5plugin
import numpy as np
import matplotlib.pyplot as plt
filename = "train.h5"



def convert_to_phi_eta_pt(p):
    # Extract the components of the momentum array
    E, px, py, pz = p[:,0], p[:,1], p[:,2], p[:,3]

    # Calculate the transverse momentum
    pt = np.sqrt(px**2 + py**2)

    # Calculate the magnitude of the momentum vector
    p_mag = np.sqrt(px**2 + py**2 + pz**2)

    # Calculate the rapidity
    y = np.zeros_like(E)  # Initialize y to zero
    valid = np.abs(E + pz) > 1e-6  # Check for valid values of E and pz
    y[valid] = 0.5 * np.log((E[valid] + pz[valid])/(E[valid] - pz[valid]))

    # Calculate the pseudorapidity
    eta = np.zeros_like(E)  # Initialize eta to zero
    valid = pt > 1e-6  # Check for valid values of pt
    eta[valid] = -np.log(np.tan(0.5 * np.arctan2(np.exp(-y[valid]), pt[valid])))

    # Calculate the azimuthal angle
    phi = np.arctan2(py, px)

    return phi, eta, pt/sum(pt)

def convert_to_jet_images(eta, phi, pt, n_pixels=32):
    # Define pixel grid
    eta_min, eta_max = np.min(eta), np.max(eta)
    phi_min, phi_max = np.min(phi), np.max(phi)
    eta_bins = np.linspace(eta_min, eta_max, n_pixels+1)
    phi_bins = np.linspace(phi_min, phi_max, n_pixels+1)

    # Compute 2D histogram of jet constituents
    pixel_grid, _, _ = np.histogram2d(eta, phi, bins=[eta_bins, phi_bins], weights=pt)

    # Normalize pixel intensities by total pT
    pixel_grid /= np.sum(pixel_grid)

    return pixel_grid




f = h5py.File(filename, 'r')
dataset = f['/table/table']
d1 = f.get('/table/table')
ds_arr = f['/table/table'][()]

train = []
for i in range(150000):
    a, b, c = ds_arr[i]
    b = np.reshape(b, [201, 4])
    phi, eta, pt = convert_to_phi_eta_pt(b)
    jet_image = convert_to_jet_images(eta, phi, pt)
    train.append(jet_image)

print(np.shape(train))
np.save('a.npy', train)

