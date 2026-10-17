import numpy as np
import sys

fileName = "sn_data(1).txt"
try:
    redshifts, m_effectives, errs = np.loadtxt(fileName, unpack=True, usecols=(1, 2, 3))
except FileNotFoundError:
    sys.exit(f"Error: file \"{fileName}\" not found")
print(redshifts)
