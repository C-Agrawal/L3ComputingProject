import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sys
from collections.abc import Callable

# constants
FILE_NAME = "sn_data.txt"
H0 = 75  # km/s/Mpc
SPEED_OF_LIGHT = 3e8  # m/s
MPC_IN_KM = 3.0857e19  # km
H0_HZ = H0 / MPC_IN_KM  # Hz
SMALL_Z_THRESHOLD = 0.1  # threshold for small redshift approximation
F_LAMBDA_0 = 6.61e-2  # Wm^-3 (10^7 erg/s/cm^2/Å)


# function to read data from file
def read_data(file_name) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        redshifts, m_effectives, errs = np.loadtxt(file_name, unpack=True, usecols=(1, 2, 3))
    except FileNotFoundError:
        sys.exit(f"Error: file \"{file_name}\" not found")
    return redshifts, m_effectives, errs

# function to calculate distance from redshift using approximation for small redshifts
def calculate_distance_small_z(redshifts: np.ndarray) -> np.ndarray:
    return (SPEED_OF_LIGHT / H0_HZ) * redshifts

# function to find the index of the first redshift less than SMALL_Z_THRESHOLD (8.4.2)
# ONLY WORKS FOR ORDERED LISTS
def find_small_z_index(redshifts: np.ndarray) -> int:
    for i, z in enumerate(redshifts):
        if z < SMALL_Z_THRESHOLD:
            return i
    return len(redshifts)  # all redshifts are greater than or equal to SMALL_Z_THRESHOLD

# function to calculate effective flux from effective magnitudes (8.19)
def calculate_effective_flux(m_effectives: np.ndarray) -> np.ndarray:
    return F_LAMBDA_0 * 10 ** (-m_effectives / 2.5)

# model for relation between redshift and effective magnitude for small redshifts
def model_function(redshifts: np.ndarray, *params) -> np.ndarray:
    distances = calculate_distance_small_z(redshifts)
    fluxes = params[0] / (4 * np.pi * distances ** 2)
    m_effectives = -2.5 * np.log10((1 + redshifts) * fluxes / F_LAMBDA_0)
    return m_effectives

# function to optimise a model
def optimise_model(redshifts: np.ndarray, m_effectives: np.ndarray, errs: np.ndarray, func: Callable) -> tuple[np.ndarray, np.ndarray]:
    initial_guess = [3e40]  # initial guess for parameters
    popt, pcov = sp.optimize.curve_fit(func, redshifts, m_effectives, sigma=errs, p0=initial_guess)
    return popt, pcov

def return_chi_squared(redshifts: np.ndarray, m_effectives: np.ndarray, errs: np.ndarray, popt: np.ndarray) -> float:
    residuals = m_effectives - model_function(redshifts, *popt)
    chi_squared = np.sum((residuals / errs) ** 2)
    return chi_squared

redshifts, m_effectives, errs = read_data(FILE_NAME)  # all unitless

small_z_index = find_small_z_index(redshifts)
print(f"Index of first redshift less than {SMALL_Z_THRESHOLD}: {small_z_index}")

distances_small_z = calculate_distance_small_z(redshifts[small_z_index:])  # meters
fluxes_small_z = calculate_effective_flux(m_effectives[small_z_index:])  # erg/s/cm^2/Å

popt, pcov = optimise_model(redshifts[small_z_index:], m_effectives[small_z_index:], errs[small_z_index:], model_function)
plt.figure(figsize=(10, 6))
plt.errorbar(redshifts[small_z_index:], m_effectives[small_z_index:], 
             yerr=errs[small_z_index:], fmt='x', color='black', 
             capsize=2, markeredgewidth=1)

smooth_x = np.linspace(np.min(redshifts[small_z_index:]), np.max(redshifts[small_z_index:]), 1000)
plt.plot(smooth_x, 
         model_function(smooth_x , popt), 
         color='r',
         label = rf"Optimised Model: $L_{{\mathrm{{peak}}}}$ = {popt[0]:.2e} W")

plt.legend(loc="best")
plt.ylabel("Effective Magnitude")
plt.xlabel("Redshift Factor")
plt.title("Effective Magnitude vs. Redshift")
plt.savefig("m_against_z.png")
plt.show()

chi_squared = return_chi_squared(redshifts[small_z_index:], m_effectives[small_z_index:], errs[small_z_index:], popt)
print(f"Chi-squared: {chi_squared}")
degrees_of_freedom = len(redshifts[small_z_index:]) - len(popt)
print(f"Reduced Chi-squared: {chi_squared / degrees_of_freedom}")
print(f"Optimised parameters: {popt}")