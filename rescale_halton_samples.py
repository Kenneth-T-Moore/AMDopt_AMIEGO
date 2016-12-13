""" Rescale the Halton samples so that they are ints on [0, 3]
"""

import numpy as np

raw = np.loadtxt('halton19.dat')

raw = raw*6.0/5.0
raw = np.floor(raw)

np.savetxt('halton19_int.dat', raw)
zz = raw.astype(int).tolist()

for z in zz:
    print(z)
