import numpy as np
import sys

file_name = "sn_data(1).txt"
try:
    redshifts, m_effectives, errs = np.loadtxt(file_name, unpack=True, usecols=(1, 2, 3))
except FileNotFoundError:
    sys.exit(f"Error: file \"{file_name}\" not found")
print(redshifts)
