import numpy as np

redshifts, m_effectives, errs = np.loadtxt("sn_data(1).txt", unpack=True, usecols=(1, 2, 3))
print(redshifts)
