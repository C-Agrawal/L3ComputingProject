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
SPEED_OF_LIGHT = 2.99792458e10  # cm/s
KM_PER_MPC = 3.0857e19  # km/Mpc
H0_HZ = H0 / KM_PER_MPC  # Hz
SMALL_Z_THRESHOLD = 0.1  # threshold for small redshift approximation
F_LAMBDA_0 = 6.61e-9  # erg/s/cm^2/Å


# function to read data from file
def read_data(file_name) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read data from a file and return redshifts, effective magnitudes, and errors.
    
    :param file_name: The name of the file to read.
    :returns: A tuple containing three arrays: redshifts, effective magnitudes, and errors.
    """
    try:
        redshifts, m_effectives, errs = np.loadtxt(file_name, unpack=True, usecols=(1, 2, 3))
    except FileNotFoundError:
        sys.exit(f"Error: file \"{file_name}\" not found")
    return redshifts, m_effectives, errs

# function to calculate distance in cm from redshift using approximation for small redshifts
def calculate_distance_small_z(redshifts: np.ndarray) -> np.ndarray:
    """Calculate distance in cm from redshift using approximation for small redshifts.
    
    :param redshifts: redshifts.
    :returns: distances in cm.
    """
    return (SPEED_OF_LIGHT / H0_HZ) * redshifts

def distance_intergrand(scale_factors: np.ndarray, omega_Lambda: float) -> np.ndarray:
    """Calculate integrand for distance calculation for large redshifts. see (8.13), (8.14)
    
    :param scale_factors: scale factors.
    :param omega_Lambda: The density parameter for dark energy.
    :returns: integrand values.
    """
    omega_m = 1 - omega_Lambda
    operand = omega_m * scale_factors + omega_Lambda * scale_factors ** 4 + (1 - omega_m - omega_Lambda) * scale_factors ** 2
    return 1 / (np.sqrt(operand))
    

def calculate_distance(redshifts: np.ndarray, omega_Lambda: float) -> np.ndarray:
    """Calculate distance in cm from redshift using numerical integration for large redshifts. see (8.13), (8.14)
    
    :param redshifts: redshifts.
    :param omega_Lambda: The density parameter for dark energy.
    :returns: distances in cm.
    """
    coeff = SPEED_OF_LIGHT / H0_HZ
    scale_factors = 1 / (1 + redshifts)
    sample_points = np.linspace(scale_factors, 1, 1001, axis=-1)  # 1001 sample points for integration
    integrand_values = distance_intergrand(sample_points, omega_Lambda)
    integrals = integrate.simpson(integrand_values, sample_points, axis=-1)
    return coeff * integrals

# model for relation between redshift and effective magnitude for small redshifts
def small_z_model_function(redshifts: np.ndarray, L_lambda_peak: float) -> np.ndarray:
    """Model for relation between redshift and effective magnitude for small redshifts. see (8.4.2), (8.18), (8.19)
    
    :param redshifts: redshifts.
    :param L_lambda_peak: The peak luminosity per unit wavelength.
    :returns: effective magnitudes.
    """
    distances = calculate_distance_small_z(redshifts)  # cm
    fluxes = L_lambda_peak / (4 * np.pi * distances ** 2 * (1+redshifts)**2)
    m_effectives = -2.5 * np.log10(fluxes / F_LAMBDA_0)
    return m_effectives

def all_z_model_function(redshifts: np.ndarray, omega_Lambda: float, L_lambda_peak: float) -> np.ndarray:
    """Model for relation between redshift and effective magnitude for all redshifts. see (8.14), (8.18), (8.19)
    
    :param redshifts: redshifts.
    :param omega_Lambda: The density parameter for dark energy.
    :param L_lambda_peak: The peak luminosity per unit wavelength.
    :returns: effective magnitudes.
    """
    distances = calculate_distance(redshifts, omega_Lambda=omega_Lambda)  # cm
    fluxes = L_lambda_peak / (4 * np.pi * distances ** 2 * (1+redshifts)**2)
    m_effectives = -2.5 * np.log10(fluxes / F_LAMBDA_0)
    return m_effectives

# function to calculate chi-squared value
def small_z_return_chi_squared(redshifts: np.ndarray, m_effectives: np.ndarray, errs: np.ndarray, L_lambda_peak: float) -> tuple[float, float]:
    """Calculate chi-squared value for fit from :func:`small_z_model_function`.
    
    :param redshifts: redshifts.
    :param m_effectives: effective magnitudes.
    :param errs: errors in effective magnitudes.
    :param L_lambda_peak: The peak luminosity per unit wavelength.
    :returns: tuple of [chi-squared value, reduced chi-squared value].
    """
    residuals = m_effectives - small_z_model_function(redshifts, L_lambda_peak)
    chi_squared = np.sum((residuals / errs) ** 2)
    degrees_of_freedom = len(redshifts) - 1  # one parameter fitted
    reduced_chi_squared = chi_squared / degrees_of_freedom
    return chi_squared, reduced_chi_squared

def all_z_return_chi_squared(redshifts: np.ndarray, m_effectives: np.ndarray, errs: np.ndarray, omega_Lambda: float, 
                             L_lambda_peak: float) -> tuple[float, float]:
    """Calculate chi-squared value for fit from :func:`all_z_model_function`.
    
    :param redshifts: redshifts.
    :param m_effectives: effective magnitudes.
    :param errs: errors in effective magnitudes.
    :param omega_Lambda: The density parameter for dark energy.
    :param L_lambda_peak: The peak luminosity per unit wavelength.
    :returns: tuple of [chi-squared value, reduced chi-squared value].
    """
    residuals = m_effectives - all_z_model_function(redshifts, omega_Lambda, L_lambda_peak)
    chi_squared = np.sum((residuals / errs) ** 2)
    degrees_of_freedom = len(redshifts) - 2  # two parameters fitted
    reduced_chi_squared = chi_squared / degrees_of_freedom
    return chi_squared, reduced_chi_squared

def display_result(result: float, error: float, chi_squared: float, reduced_chi_squared: float, unit: str, result_name: str) -> None:
    """Display the result of the optimisation and chi-squared values.
    
    :param result: The optimised result.
    :param error: The error in the optimised result.
    :param chi_squared: The chi-squared value.
    :param reduced_chi_squared: The reduced chi-squared value.
    :param unit: The unit of the result.
    :param result_name: The name of the result.
    """

    print(f"Optimised {result_name}: {result:.4e} ± {error:.4e} {unit}")
    print(f"Chi-squared: {chi_squared}")
    print(f"Reduced Chi-squared: {reduced_chi_squared}")
    

def plot_small_z_model(redshifts: np.ndarray, m_effectives: np.ndarray, errs: np.ndarray, L_lambda_peak: float, funct: Callable, file_name: str) -> None:
    """Plot the optimised model from :func:`small_z_model_function` against the data for small redshifts.
    
    :param redshifts: redshifts.
    :param m_effectives: effective magnitudes.
    :param errs: errors in effective magnitudes.
    :param L_lambda_peak: The peak luminosity per unit wavelength.
    :param funct: The model function to plot.
    """
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
    plt.savefig(file_name)
    plt.show()

def plot_all_z_model(redshifts: np.ndarray, m_effectives: np.ndarray, errs: np.ndarray, omega_Lambda: float, 
                     L_lambda_peak: float, funct: Callable, file_name: str) -> None:
    """Plot the optimised model from :func:`all_z_model_function` against the data for all redshifts.

    :param redshifts: redshifts.
    :param m_effectives: effective magnitudes.
    :param errs: errors in effective magnitudes.
    :param omega_Lambda: The density parameter for dark energy.
    :param L_lambda_peak: The peak luminosity per unit wavelength.
    :param funct: The model function to plot.
    """
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
    plt.savefig(file_name)
    plt.show()



def main()-> None:
    """Main function to read data, find optimised L_lambda_peak and omega_Lambda, and plot the results."""

    # read and initialise data
    redshifts, m_effectives, errs = read_data(FILE_NAME)  # all unitless
    small_z = redshifts < SMALL_Z_THRESHOLD
    redshifts_small, m_effectives_small, errs_small = redshifts[small_z], m_effectives[small_z], errs[small_z]


    # optimise L_lambda_peak using small redshifts
    bounds = (1e30, np.inf)  # ensure parameters are positive
    initial_guess = [3e40]  # initial guess for L_lambda_peak
    popt, cov = optimize.curve_fit(small_z_model_function, redshifts_small, m_effectives_small, sigma=errs_small, p0=initial_guess, bounds=bounds)

    
    L_lambda_peak = popt[0]  # ergs/s/Å
    err_L_lambda_peak = np.sqrt(np.diag(cov))[0]  # ergs/s/Å

    chi_squared, reduced_chi_squared = small_z_return_chi_squared(redshifts_small, m_effectives_small, errs_small, popt[0])
    display_result(L_lambda_peak, err_L_lambda_peak, chi_squared, reduced_chi_squared, "ergs/s/Å", "L_λ_peak")

    plot_small_z_model(redshifts_small, m_effectives_small, errs_small, popt[0], funct=small_z_model_function, file_name="m_against_z_small.png")

    # optimise omega_Lambda using all redshifts (method 1)
    bounds = (0, 1)  # ensure omega_Lambda is between 0 and 1
    initial_guess = [0.5]  # initial guess for omega_Lambda
    popt, cov = optimize.curve_fit(lambda redshifts, omega_Lambda: all_z_model_function(redshifts, omega_Lambda, L_lambda_peak), redshifts, 
                                   m_effectives, sigma=errs, p0=initial_guess, bounds=bounds)

    omega_Lambda = popt[0]  # no units
    err_omega_Lambda = np.sqrt(np.diag(cov))[0]  # no units

    chi_squared, reduced_chi_squared = all_z_return_chi_squared(redshifts, m_effectives, errs, popt[0], L_lambda_peak)
    display_result(omega_Lambda, err_omega_Lambda, chi_squared, reduced_chi_squared, "(no units)", "Omega_Lambda")

    plot_all_z_model(redshifts, m_effectives, errs, popt[0], L_lambda_peak, funct=all_z_model_function, file_name="m_against_z.png")

    # sampling chi-squared
    omega_Lambda_samples = np.linspace(0, 1, 100)
    reduced_chi_squared_samples = []
    for omega in omega_Lambda_samples:
        _, reduced_chi_squared = all_z_return_chi_squared(redshifts, m_effectives, errs, omega, L_lambda_peak)
        reduced_chi_squared_samples.append(reduced_chi_squared)
    plt.figure(figsize=(10, 6))
    plt.plot(omega_Lambda_samples, reduced_chi_squared_samples, color='g')
    plt.xlabel("Omega_Lambda")
    plt.ylabel("Reduced Chi-squared")
    plt.title("Reduced Chi-squared vs. Omega_Lambda")
    plt.savefig("reduced_chi_squared_vs_omega_lambda.png")
    plt.show()
    print(f"Minimum Reduced Chi-squared: {min(reduced_chi_squared_samples)} at Omega_Lambda = {omega_Lambda_samples[np.argmin(reduced_chi_squared_samples)]}")

if __name__ == "__main__":
    main()
