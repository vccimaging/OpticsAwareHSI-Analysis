import sys
sys.path.append('..')
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
from . import utils

def show_color_difference(rgb1, rgb2, norm=255, gamma=2.2):
    """
    Display the difference in two color images.

    Args:
        rgb1, rgb2: Two RGB images of shape [M, N, C].
        gamma: Gamma correction.
    """
    err = np.abs(rgb2/norm - rgb1/norm)
    err_disp = err/err.max()

    plt.figure(figsize=(12,8))
    ax0 = plt.subplot(1,3,1)
    ax0.imshow(rgb1)
    ax0.axis('off')
    ax0.set_title('RGB 1')
    ax1 = plt.subplot(1,3,2)
    ax1.imshow(rgb2)
    ax1.axis('off')
    ax1.set_title('RGB 2')
    ax2 = plt.subplot(1,3,3)
    ax2.imshow(err_disp**(1/gamma))
    ax2.axis('off')
    ax2.set_title(f"max = {err.max():.4f}")
    plt.show()

def show_rgb_from_hsi(hsi, srf, shape=None, order='HWC'):
    """
    Display the hyperspectral datacube by projecting it into an RGB image.

    Args:
        hsi: A numpy array for the hyperspectral datacube of shape [H, W, C] or [C, H, W].
        srf: A numpy array for the spectral response function of shape [C, 3].
        shape: An integer or tuple for the target shape for display.
        order: A string that indicates the order of channels. Valid options are 'HWC' and 'CHW'.
    """
    if order == 'HWC':
        _hsi = hsi
    elif order == 'CHW':
        _hsi = np.transpose(hsi, (1,2,0))
    else:
        raise ValueError(f"Invalid order {order}. Valid options are HWC or CHW.")

    rgb = utils.projectHSItoRGB_np(_hsi, srf, True)
    rgb = rgb / rgb.max()
    if shape is not None:
        if np.isscalar(shape):
            shape = (shape, shape)
        rgb = utils.crop_or_pad_np(rgb, shape)

    plt.figure()
    plt.imshow(rgb)
    plt.show()

def show_spectral_grid(hsi, wavelengths, srf, cols=5, aspect=1.05, shape=None, order='HWC', colorspace='linear', normalize=False):
    """
    Display the hyperspectral datacube in color images on a grid.

    Args:
        hsi: A numpy array for hyperspectral datacube. Shape is [M, N, C].
        wavelengths: A numpy array for the wavelengths.
        A numpy array for the spectral response function of shape [C, 3].
        cols: Number of columns of the grid.
        aspect: Aspect ratio.
        shape: An integer or tuple for the target shape for display.
        order: A string that indicates the order of channels. Valid options are 'HWC' and 'CHW'.
    """
    if order == 'HWC':
        _hsi = hsi
    elif order == 'CHW':
        _hsi = np.transpose(hsi, (1,2,0))
    else:
        raise ValueError(f"Invalid order {order}. Valid options are HWC or CHW.")
    
    H, W, C = _hsi.shape
    rows = int(np.ceil(C/cols))
    fig_width = 10
    fig_height = fig_width * rows / cols * aspect
    plt.figure(figsize=(fig_width, fig_height))
    for i in range(rows):
        for j in range(cols):
            index = cols*i+j
            if index < C:
                PSF_rgb = singleband2rgb(wavelengths[index], _hsi[:,:,index], wavelengths, srf, colorspace=colorspace, normalize=normalize)
                if shape is not None:
                    if np.isscalar(shape):
                        shape = (shape, shape)
                    PSF_rgb = utils.crop_or_pad_np(PSF_rgb, shape)
                ax = plt.subplot(rows, cols, index+1)
                ax.imshow(PSF_rgb)
                ax.axis('scaled')
                ax.axis('off')
                ax.set_title(f"{int(wavelengths[index]*1e3):03d} nm")          
    plt.show()

def singleband2rgb(w, image, wavelengths, srf, colorspace='linear', normalize=False):
    """Converts a single band spectral image to a color image.

    Args:
        w: A scalar for the wavelength of the single band image.
        image: A 2D array for the single band image.
        wavelengths: A 1D array for the wavelengths where the SRF is defined.
        srf: A 2D tensor for the spectral response function in R, G and B channels.

    Returns:
        An RGB color image.
    """

    if len(np.isclose(w, wavelengths).nonzero()) == 0:
        raise ValueError("The single band wavelength does not match the SRF.")
    
    index = np.squeeze(np.isclose(w, wavelengths).nonzero())
    rgb = np.expand_dims(image, -1) * srf[index]

    if normalize:
        rgb = rgb/rgb.max()
    
    if colorspace == 'srgb':
        rgb = lin2srgb(rgb)

    return rgb

def lin2srgb(x):
    """Converts an linear RGB image to an sRGB image.

    Args:
        x: A numpy array for the linear RGB image.

    Returns:
        An sRGB color image.
    """

    gamma = 1/2.4
    a = 1.055
    b = -0.055
    c = 12.92
    d = 0.0031308

    y = np.zeros_like(x)
    in_sign = -2 * (x < 0) + 1
    x = np.abs(x)
    y = np.where(x < d, c*x, a * np.exp(gamma*np.log(x)) + b)

    return y*in_sign