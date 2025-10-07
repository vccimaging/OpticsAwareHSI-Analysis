"""
Note 1: We adopt the following parameters from the NTIRE2022_spectral repository by Arad Boaz.
Source: https://github.com/boazarad/NTIRE2022_spectral/blob/main/Conf.py
Accessed on: 2025-04-23
"""

import numpy as np

RGB_FILTER_CSV = '../resources/RGB_Camera_QE.csv'
MOSAIC_FILTER_CSV = '../resources/MS_Camera_QE.csv'
ANALOG_CHANNEL_GAIN = np.array([2.2933984, 1, 1.62308182])

TYPICAL_SCENE_REFLECTIVITY = 0.15
MAX_VAL_8_BIT = (2 ** 8 - 1)
MAX_VAL_12_BIT = (2 ** 12 - 1)

"""
Note 2: We add the following global parameters for the analysis in our paper.

Author: Qiang Fu
Email: qiang.fu@kaust.edu.sa
"""

# wavelength range
lambda1 = 0.4
lambda2 = 0.7
dlambda = 0.01
WAVELENGTHS = np.arange(lambda1, lambda2+dlambda, dlambda)
# original spatial dimensions
HEIGHT, WIDTH = 1392, 1300