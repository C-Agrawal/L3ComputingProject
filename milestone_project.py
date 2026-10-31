import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sys

# constants
FILE_NAME = "sn_data.txt"
H0 = 75  # km/s/Mpc
SPEED_OF_LIGHT = 3e8  # m/s
SMALL_Z_THRESHOLD = 0.1  # threshold for small redshift approximation
F_LAMBDA = 6.61e-9  # erg/s/cm^2/Å. to get SI, multiply by 1e7

# function to read data from file
def read_data(file_name) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        redshifts, m_effectives, errs = np.loadtxt(file_name, unpack=True, usecols=(1, 2, 3))
    except FileNotFoundError:
        sys.exit(f"Error: file \"{file_name}\" not found")
    return redshifts, m_effectives, errs

# function to calculate distance from redshift using approximation for small redshifts
def calculate_distance_small_z(redshifts: np.ndarray) -> np.ndarray:
    return (SPEED_OF_LIGHT / H0) * redshifts

# function to find the index of the first redshift less than SMALL_Z_THRESHOLD (8.4.2)
# ONLY WORKS FOR ORDERED LISTS
def find_small_z_index(redshifts: np.ndarray) -> int:
    for i, z in enumerate(redshifts):
        if z < SMALL_Z_THRESHOLD:
            return i
    return len(redshifts)  # all redshifts are greater than or equal to SMALL_Z_THRESHOLD

# function to calculate effective flux from effective magnitudes (8.19)
def calculate_effective_flux(m_effectives: np.ndarray) -> np.ndarray:
    return F_LAMBDA * 10 ** (-m_effectives / 2.5)

redshifts, m_effectives, errs = read_data(FILE_NAME)  # all unitless

small_z_index = find_small_z_index(redshifts)

distances_small_z = calculate_distance_small_z(redshifts[small_z_index:])  # meters
fluxes_small_z = calculate_effective_flux(m_effectives[small_z_index:])  # erg/s/cm^2/Å

plt.plot(distances_small_z, fluxes_small_z, 'o')
plt.savefig("flux_vs_distance.png")