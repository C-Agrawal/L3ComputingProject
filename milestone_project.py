import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sys
from collections.abc import Callable

# all units are in cgs unless otherwise stated

# constants
FILE_NAME = "sn_data.txt"
H0 = 75  # km/s/Mpc
SPEED_OF_LIGHT = 3e10  # cm/s
KM_PER_MPC = 3.0857e19  # km/Mpc
H0_HZ = H0 / KM_PER_MPC  # Hz
SMALL_Z_THRESHOLD = 0.1  # threshold for small redshift approximation
F_LAMBDA_0 = 6.61e-9  # erg/s/cm^2/Å


# function to read data from file
def read_data(file_name) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        redshifts, m_effectives, errs = np.loadtxt(file_name, unpack=True, usecols=(1, 2, 3))
    except FileNotFoundError:
        sys.exit(f"Error: file \"{file_name}\" not found")
    return redshifts, m_effectives, errs

# function to calculate distance in cm from redshift using approximation for small redshifts
def calculate_distance_small_z(redshifts: np.ndarray) -> np.ndarray:
    return (SPEED_OF_LIGHT / H0_HZ) * redshifts

# function to calculate effective flux from effective magnitudes (8.19)
def calculate_effective_flux(m_effectives: np.ndarray) -> np.ndarray:
    return F_LAMBDA_0 * 10 ** (-m_effectives / 2.5)

# model for relation between redshift and effective magnitude for small redshifts
def small_z_model_function(redshifts: np.ndarray, *params) -> np.ndarray:
    distances = calculate_distance_small_z(redshifts)  # cm
    fluxes = params[0] / (4 * np.pi * distances ** 2)
    m_effectives = -2.5 * np.log10((1 + redshifts) * fluxes / F_LAMBDA_0)
    return m_effectives

# function to optimise a model
def optimise_model(redshifts: np.ndarray, m_effectives: np.ndarray, errs: np.ndarray, func: Callable, 
                   initial_guess: list) -> tuple[np.ndarray, np.ndarray]:
    bounds = (1e30, np.inf)  # ensure parameters are positive
    popt, pcov = sp.optimize.curve_fit(func, redshifts, m_effectives, sigma=errs, p0=initial_guess, bounds=bounds)
    return popt, pcov

# function to calculate chi-squared value
def return_chi_squared(redshifts: np.ndarray, m_effectives: np.ndarray, errs: np.ndarray, popt: np.ndarray) -> float:
    residuals = m_effectives - small_z_model_function(redshifts, *popt)
    chi_squared = np.sum((residuals / errs) ** 2)
    return chi_squared

def plot_small_z_model(redshifts: np.ndarray, m_effectives: np.ndarray, errs: np.ndarray, *popt: np.ndarray, funct: Callable) -> None:
    plt.figure(figsize=(10, 6))
    plt.errorbar(redshifts, m_effectives, 
                 yerr=errs, fmt='x', color='black', 
                 capsize=2, markeredgewidth=1)

    smooth_x = np.linspace(np.min(redshifts), np.max(redshifts), 1000)
    plt.plot(smooth_x, 
             funct(smooth_x , *popt), 
             color='r',
             label = rf"Optimised Model: $L_{{\mathrm{{\lambda,peak}}}}$ = {popt[0]:.2e} ergs/s/Å")

    plt.legend(loc="best")
    plt.ylabel("Effective Magnitude")
    plt.xlabel("Redshift Factor")
    plt.title("Effective Magnitude vs. Redshift")
    plt.savefig("m_against_z.png")
    plt.show()

def main()-> None:
    redshifts, m_effectives, errs = read_data(FILE_NAME)  # all unitless
    small_z = redshifts < SMALL_Z_THRESHOLD
    redshifts_small, m_effectives_small, errs_small = redshifts[small_z], m_effectives[small_z], errs[small_z]

    popt, cov = optimise_model(redshifts_small, m_effectives_small, errs_small, small_z_model_function, initial_guess=[3e40])

    plot_small_z_model(redshifts_small, m_effectives_small, errs_small, *popt, funct=small_z_model_function)

    L_lambda_peak = popt[0]  # ergs/s/Å
    err_L_lambda_peak = np.sqrt(np.diag(cov))[0]  # ergs/s/Å

    chi_squared = return_chi_squared(redshifts_small, m_effectives_small, errs_small, popt)
    degrees_of_freedom = len(redshifts_small) - len(popt)
    reduced_chi_squared = chi_squared / degrees_of_freedom

    print(f"Chi-squared: {chi_squared}")
    print(f"Reduced Chi-squared: {reduced_chi_squared}")
    print(f"L_lambda_peak: {L_lambda_peak:.2e} ± {err_L_lambda_peak:.2e} ergs/s/Å")

if __name__ == "__main__":
    main()
