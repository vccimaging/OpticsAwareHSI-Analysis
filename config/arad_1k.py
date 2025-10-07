"""
Note 1: We adopt the following parameters from the NTIRE2022_spectral repository by Arad Boaz.
Source: https://github.com/boazarad/NTIRE2022_spectral/blob/main/Conf.py
Accessed on: 2025-04-23
"""

# Global Constants
import numpy as np

# Attempt to load private configuration file with confidential competition parameters
try:
    from PrivateConf import *
except ModuleNotFoundError:
    # Sample parameters which are similar to the confidential competition parameters BUT NOT IDENTICAL!
    NOISE = 750
    JPEG_QUALITY = 65

RGB_FILTER_CSV = '../resources/RGB_Camera_QE.csv'
MOSAIC_FILTER_CSV = '../resources/MS_Camera_QE.csv'
ANALOG_CHANNEL_GAIN = np.array([2.2933984, 1, 1.62308182])

TYPICAL_SCENE_REFLECTIVITY = 0.18
MAX_VAL_8_BIT = (2 ** 8 - 1)
MAX_VAL_12_BIT = (2 ** 12 - 1)

SIZE = 512
QUARTER = SIZE // 4
CROP = np.s_[QUARTER:-QUARTER, QUARTER:-QUARTER]  # keep only the center 50% of the image

SUBMISSION_SIZE_LIMIT = 5*10**8  # (500MB)

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
HEIGHT = 482
WIDTH = 512