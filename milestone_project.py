import numpy as np
import scipy as sp
from scipy import integrate, optimize
import matplotlib.pyplot as plt
import sys
from collections.abc import Callable

# all units are in cgs unless otherwise stated
# k=0 is hardcoded. Universe is flat. Might change later

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

def distance_intergrand(scale_factors: np.ndarray, omega_Lambda: float) -> np.ndarray:
    omega_m = 1 - omega_Lambda
    operand = omega_m * scale_factors + omega_Lambda * scale_factors ** 4 + (1 - omega_m - omega_Lambda) * scale_factors ** 2
    return 1 / (np.sqrt(operand))
    

def calculate_distance(redshifts: np.ndarray, omega_Lambda: float) -> np.ndarray:
    coeff = SPEED_OF_LIGHT / H0_HZ
    scale_factors = 1 / (1 + redshifts)
    sample_points = np.linspace(scale_factors, 1, 201, axis=-1)  # 201 sample points for integration
    integrand_values = distance_intergrand(sample_points, omega_Lambda)
    integrals = integrate.simpson(integrand_values, sample_points, axis=-1)
    return coeff * integrals

# function to calculate effective flux from effective magnitudes (8.19)
def calculate_effective_flux(m_effectives: np.ndarray) -> np.ndarray:
    return F_LAMBDA_0 * 10 ** (-m_effectives / 2.5)

# model for relation between redshift and effective magnitude for small redshifts
def small_z_model_function(redshifts: np.ndarray, *params: np.ndarray) -> np.ndarray:
    distances = calculate_distance_small_z(redshifts)  # cm
    fluxes = params[0] / (4 * np.pi * distances ** 2)
    m_effectives = -2.5 * np.log10((1 + redshifts) * fluxes / F_LAMBDA_0)
    return m_effectives

def all_z_model_function(redshifts: np.ndarray, omega_Lambda: float, L_lambda_peak: float) -> np.ndarray:
    distances = calculate_distance(redshifts, omega_Lambda=omega_Lambda)  # cm
    fluxes = L_lambda_peak / (4 * np.pi * distances ** 2 * (1+redshifts)**2)
    m_effectives = -2.5 * np.log10((1 + redshifts) * fluxes / F_LAMBDA_0)
    return m_effectives

# function to calculate chi-squared value
def return_chi_squared(redshifts: np.ndarray, m_effectives: np.ndarray, errs: np.ndarray, L_lambda_peak: float) -> float:
    residuals = m_effectives - small_z_model_function(redshifts, L_lambda_peak)
    chi_squared = np.sum((residuals / errs) ** 2)
    return chi_squared

def plot_small_z_model(redshifts: np.ndarray, m_effectives: np.ndarray, errs: np.ndarray, L_lambda_peak: float, funct: Callable) -> None:
    plt.figure(figsize=(10, 6))
    plt.errorbar(redshifts, m_effectives, 
                 yerr=errs, fmt='x', color='black', 
                 capsize=2, markeredgewidth=1)

    smooth_x = np.linspace(np.min(redshifts), np.max(redshifts), 1000)
    plt.plot(smooth_x, 
             funct(smooth_x, L_lambda_peak), 
             color='r',
             label = rf"Optimised Model: $L_{{\mathrm{{\lambda,peak}}}}$ = {L_lambda_peak:.2e} ergs/s/Å")

    plt.legend(loc="best")
    plt.ylabel("Effective Magnitude")
    plt.xlabel("Redshift Factor")
    plt.title("Effective Magnitude vs. Redshift")
    plt.savefig("m_against_z.png")
    plt.show()

def plot_all_z_model(redshifts: np.ndarray, m_effectives: np.ndarray, errs: np.ndarray, omega_Lambda: float, 
                     L_lambda_peak: float, funct: Callable) -> None:
    plt.figure(figsize=(10, 6))
    plt.errorbar(redshifts, m_effectives, 
                    yerr=errs, fmt='x', color='black', 
                    capsize=2, markeredgewidth=1)

    smooth_x = np.linspace(np.min(redshifts), np.max(redshifts), 1000)
    plt.plot(smooth_x, 
                funct(smooth_x ,omega_Lambda, L_lambda_peak), 
                color='b',
                label = rf"Optimised Model: $\Omega_{{\mathrm{{\Lambda}}}}$ = {omega_Lambda:.2f}")

    plt.legend(loc="best")
    plt.ylabel("Effective Magnitude")
    plt.xlabel("Redshift Factor")
    plt.title("Effective Magnitude vs. Redshift")
    plt.savefig("m_against_z_full.png")
    plt.show()
    pass  # Placeholder for future implementation of plotting for all redshifts

def main()-> None:
    redshifts, m_effectives, errs = read_data(FILE_NAME)  # all unitless
    small_z = redshifts < SMALL_Z_THRESHOLD
    redshifts_small, m_effectives_small, errs_small = redshifts[small_z], m_effectives[small_z], errs[small_z]

    bounds = (1e30, np.inf)  # ensure parameters are positive
    initial_guess = [3e40]  # initial guess for L_lambda_peak
    popt, cov = optimize.curve_fit(small_z_model_function, redshifts_small, m_effectives_small, sigma=errs_small, p0=initial_guess, bounds=bounds)

    plot_small_z_model(redshifts_small, m_effectives_small, errs_small, popt[0], funct=small_z_model_function)

    L_lambda_peak = popt[0]  # ergs/s/Å
    err_L_lambda_peak = np.sqrt(np.diag(cov))[0]  # ergs/s/Å

    chi_squared = return_chi_squared(redshifts_small, m_effectives_small, errs_small, popt[0])
    degrees_of_freedom = len(redshifts_small) - len(popt)
    reduced_chi_squared = chi_squared / degrees_of_freedom

    print(f"Chi-squared: {chi_squared}")
    print(f"Reduced Chi-squared: {reduced_chi_squared}")
    print(f"L_lambda_peak: {L_lambda_peak:.2e} ± {err_L_lambda_peak:.2e} ergs/s/Å")

    bounds = (0, 1)  # ensure parameters are positive
    initial_guess = [0.5]  # initial guess for omega_Lambda

    popt, cov = optimize.curve_fit(lambda redshifts, omega_Lambda: all_z_model_function(redshifts, omega_Lambda, L_lambda_peak), redshifts, 
                                   m_effectives, sigma=errs, p0=initial_guess, bounds=bounds)


    plot_all_z_model(redshifts, m_effectives, errs, popt[0], L_lambda_peak, funct=all_z_model_function)
    print(f"Optimized Omega_Lambda: {popt[0]:.4f} ± {np.sqrt(np.diag(cov))[0]:.4f}")
if __name__ == "__main__":
    main()
