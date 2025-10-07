import os
import sys
sys.path.append('.')
sys.path.append('..')

import numpy as np
from scipy.special import j1
from scipy.io import loadmat
import h5py
import hdf5storage

import torch
import torch.nn.functional as F
import torch.fft as fft

EPSILON = 1e-10

def str2bool(v):
    return v.lower() in ('yes', 'y', 'true', 't', '1')

def load_srf(file_path, dtype=np.float32):
    """
    Load the sensor spectral response function as a numpy array with shape [C, 3].

    Args:
        file_path: Path to the sensor data in .npz format.
        dtype: Data type. Default to np.float32.

    Returns:
        srf: Spectral response function as a numpy array of shape[C, 3].
    """
    data = np.load(file_path)
    srf = np.vstack((data['R'], data['G'], data['B'])).T

    return srf.astype(dtype)

def load_hsi_mat(file_path, name='cube', transpose=True, dtype=np.float32):
    """
    Load a hyperspectral datacube as a numpy array.

    Args:
        file_path: Path to the mat file for hyperspectral datacube.
        name: Variable name for the data. Default as "cube".
        transpose: Reverse the order of dimensions if True.
        Note that the matlab compatible mat file has a shape of [C, W, H].
        Turn transpose on to make sure the loaded file is [H, W, C].

    Returns:
        hsi: A numpy array for the hyperspectral datacube.
    """
    with h5py.File(file_path, 'r') as mat:
        hsi = np.array(mat[name])
        if transpose:
            hsi = hsi.T
    return hsi.astype(dtype)

def save_hsi_mat(file_path, hsi, bands=None):
    """
    Save a hyperspectral datacube in Matlab HDF5 format.

    Args:
        file_path: Path to the mat file for hyperspectral datacube.
        hsi: A numpy array for the hyperspectral datacube.
        bands: A numpy array for the spectral bands if provided.
    """
    if os.path.exists(file_path):
        os.remove(file_path)

    # when saving with matlab_compatible=True, the saved data shape is reversed
    # e.g., if hsi is [m, n, k], the saved mat file shap is [k, n, m].    
    # hdf5storage.write(
    #     {u'cube': hsi, u'bands': bands}, 
    #     '.', 
    #     file_path, 
    #     matlab_compatible=True
    # )
    hdf5storage.savemat(
        file_path,
        {u'cube': hsi, u'bands': bands},
        format='7.3',
        store_python_metadata=True,
        matlab_compatible=True
    )

# numpy utilities
def projectHSItoRGB_np(hsi, srf, clip_negative=True):
    """
    Project a hyperspectral image to an RGB image. No exposure settings are considered.

    Args:
        hsi: An array for the hyperspectral datacube with shape of [H, W, C]. 
        srf: An array for the camera spectral response function with shape of [C, 3].
        clip_negative: Clip negative values if True; keep data intact otherwise.

    Returns:
        rgb: An arry for the resulting RGB image with shape [H, W, 3].
    """
    rgb = np.matmul(hsi, srf)
    if clip_negative:
        rgb = rgb.clip(0, None)

    return rgb

def add_noise_np(signal, noise_type='poisson', npe=0, divFactorTo_1PE=1, seed=0):
    """
    Add noise to input signal, depending on the noise type.

    Args:
        signal: A numpy array for the input signal. Default units: [Npe]
        noise_type: Type of noise. Could be either Poisson noise or Gaussian noise.
        npe: A parameter to control the noise level. 0 means no shot noise.
        divFactorTo_1PE: A factor to convert the input signal to Npe units. Default 1.
        seed: Random seed for the noise generation.

    Returns:
        noisy_signal: Noisy signal with the prescribed noise.
    """
    if npe == 0:
        return signal

    scale = npe / divFactorTo_1PE
    np.random.seed(seed=seed)
    if noise_type == 'poisson':
        noisy_signal = np.random.poisson(signal.clip(0, None) * scale)
        noisy_signal = noisy_signal / scale
    elif noise_type == 'gaussian':
        gauss = np.random.normal(0, scale, signal.shape)
        noisy_signal = signal + gauss
    else:
        raise ValueError(f"Unknown noise type {noise_type}.")
    
    return noisy_signal.clip(0, None)

def isp_np(image, config, bit=8):
    """
    Apply a simple ISP to the input image. Currently, only automatic exposure 
    and 8/12-bit quantization are considered.

    Args:
        image: A numpy array for the input image signal.
        config: Configuration for the camera ISP.
        bit: Bit depth of the output image.

    Returns:
        image_isp: A digital image after applying the ISP.
    """
    if bit == 8:
        max_val = config.MAX_VAL_8_BIT
    elif bit == 12:
        max_val = config.MAX_VAL_12_BIT
    else:
        raise ValueError(f"Bit depth {bit} is not supported")
    # automatic exposure
    image_isp = image * (config.TYPICAL_SCENE_REFLECTIVITY / image.mean()) * max_val
    # quantization
    image_isp = image_isp.clip(0, max_val).astype(np.uint8)

    return image_isp

def crop_or_pad_np(in_array, shape=512):
    """
    Crop or pad the input array in the first two dimensions to match the required shape.

    Args:
        in_array: An input numpy array with shape [H, W, C].
        shape: The required output shape. Should be a scalar or a tuple of two.

    Returns:
        out_array: Cropped or padded numpy array.
    """
    
    if np.isscalar(shape):
        shape = (shape, shape)
    elif len(shape) > 2:
        raise ValueError(f"Shape should be a scalar or tuple of two, but got {len(shape)}.")
    
    if in_array.ndim == 2:
        in_array = np.expand_dims(in_array, -1)
    elif in_array.ndim < 1 or in_array.ndim > 3:
        raise ValueError(f"Input array dimension ({in_array.ndim}) out of range [2, 3].")
    
    refH, refW = shape
    inH, inW = in_array.shape[:-1]
    
    out_array = in_array.copy()
    # rows
    if inH > refH:
        # crop rows
        padH = (inH - refH) // 2
        out_array = out_array[padH:-(inH-refH-padH), :, :]
    elif inH < refH:
        # pad rows (top and bottom)
        padH = (refH - inH) // 2
        out_array = np.pad(out_array, ((padH, refH-inH-padH), (0, 0), (0, 0)), 'constant', constant_values=0)
    # columns
    if inW > refW:
        # crop columns
        padW = (inW - refW) // 2
        out_array = out_array[:, padW:-(inW-refW-padW), :]
    elif inW < refW:
        # pad columns (left and right)
        padW = (refW - inW) // 2
        out_array = np.pad(out_array, ((0, 0), (padW, refW-inW-padW), (0, 0)), 'constant', constant_values=0)
    
    return out_array.squeeze()

def conv2D_np(img, psf):
    """
    Apply 2D convolution between the input image and psf.

    Args:
        img: A numpy array for the input image of shape [C, H, W].
        psf: A numpy array for the PSF of shape [C, M, N].

    Returns:
        convolved: A numpy array for the convolved image of shape [C, H, W].
    """
    # normalize the PSF in each channel
    psf = psf / np.sum(psf, axis=(-2, -1), keepdims=True)
    # process in the Fourier domain 
    convolved = np.fft.ifftshift(np.fft.ifft2(np.fft.fft2(np.fft.fftshift(img)) * np.fft.fft2(np.fft.fftshift(psf)))).real
    
    return convolved

def image_formation_np(
    hsi, srf, config, psf=None, 
    noise_type='poisson', npe=0, divFactorTo_1PE=1, bit=8,
    seed=0,
):
    """
    Perform the image formation model to generate an RGB image from a hyperspectral datacube.

    Args:
        hsi: A numpy array for the hyperspectral datacube, with shape [H, W, C].
        srf: Spectral response function of the sensor, with shape [C, 3].
        config: Configuration for the imaging process.
        psf: A numpy array for the spectral point spread function, with shape [H, W, C].
        npe: A parameter to control the noise level. Defaul to 0 (no noise).
        divFactorTo_1PE: A factor to convert the input signal to Npe units. Default 1.
        bit: Bit depth of the RGB image.
        seed: Random seed for the noise generation.

    Returns:
        rgb: A numpy array for the RGB image, with shape [H, W, 3].
    """    
    if psf is not None:
        # image formation by 2D convolution
        psf = crop_or_pad_np(psf, hsi.shape[:2])
        # transpose before convolution
        _hsi = conv2D_np(np.transpose(hsi, (2,0,1)), np.transpose(psf, (2,0,1)))
        # transpose back
        _hsi = np.transpose(_hsi, (1,2,0))
    else:
        _hsi = hsi

    # 1. Project the hyperspectral datacube to RGB
    rgb = projectHSItoRGB_np(_hsi, srf, clip_negative=True)
    # 2. Add noise
    rgb = add_noise_np(rgb, noise_type=noise_type, npe=npe, divFactorTo_1PE=divFactorTo_1PE, seed=seed)
    # 3. Apply ISP
    rgb = isp_np(rgb, config, bit=bit)

    return rgb    

def convert_hsi_to_rgb_np(hsi, srf, config):
    """
    Convert a hyperspectral datacube to an RGB image according to the configuration.
    This is equivalent to do a noiseless image formation without any PSF.

    Args:
        hsi: A numpy array for the hyperspectral datacube, with shape [H, W, C].
        srf: Spectral response function of the sensor, with shape [C, 3].
        config: Configuration for the imaging process.

    Returns:
        rgb: A numpy array for the RGB image, with shape [H, W, 3].
    """

    return image_formation_np(hsi, srf, config, psf=None, npe=0)

def generate_metamer_np(hsi, srf, alpha=0.0):
    """
    Generate metamer HSI.

    Args:
        hsi: A numpy array for the hyperspectral datacube, with shape [m, n, k].
        srf: A numpy array for the spectral response function, with shape [k, 3].
        alpha: A parameter to control the metameric black term. Default: 0.0.

    Returns:
        metamer: A metamer using the metameric black method, with shape [m, n, k].
    """
    m, n, k = hsi.shape
    S = np.transpose(np.reshape(hsi, (-1, k)), (1, 0)) # k x mn
    C = srf # k x 3
    R = C @ np.linalg.inv(C.T @ C) @ C.T # k x k
    I = np.eye(R.shape[0])
    # fundamental metamer
    S0 = R @ S
    # metameric black
    Sb = (I - R) @ S
    # augmentation
    Sx = S0 + alpha * Sb
    # clip negative values
    Sx = np.clip(Sx, 0.0, None)
    # reshape
    metamer = np.transpose(np.reshape(Sx, (k, m, n)), (1, 2, 0))

    return metamer

# torch utilities
class MetamerAugment(object):
    """
    Augment the hyperspectral image with metameric black.

    Args:
        srf: the spectral response function of the sensor.
    """

    def __init__(self, srf, alpha=0.0):
        self.srf = srf
        self.alpha = alpha

    def __call__(self, hsi):
        """
        Args:
            hsi: hyperspectral image with shape [b, k, m, n].
        """
        if hsi.ndim == 3:
            hsi = hsi.unsqueeze(0)
        b, k, m, n = hsi.shape
        S = torch.reshape(hsi, (b, k, -1)) # b x k x mn
        # compute the R matrix
        C = self.srf # k x 3
        R = C @ torch.linalg.inv(C.T @ C) @ (C.T) # k x k
        I = torch.eye(R.shape[0]).to(S.device)
        # decompose the hsi into fundamental metamer and metameric black
        # fundamental metamer
        S0 = R @ S
        # metameric black
        Sb = (I - R) @ S
        # augment the original hsi with metameric black
        Sx = S0 + self.alpha * Sb
        # clamp negative values
        Sx = torch.clamp(Sx, min=0.0)

        return Sx.reshape(b, k, m, n)
    
    def __repr__(self):
        return "Augment the hyperspectral image with metameric black."

def projectHSItoRGB(hsi, srf, clip_negative=True):
    """
    Project a hyperspectral image to an RGB image. No exposure settings are considered.

    Args:
        hsi: A tensor for the hyperspectral datacube with shape of [B, C, H, W]. 
        srf: A tensor for the camera spectral response function with shape of [C, 3].
        clip_negative: Clip negative values if True; keep data intact otherwise.

    Returns:
        rgb: A tensor for the resulting RGB image with shape [3, H, W].
    """
    rgb = torch.matmul(hsi.permute((0,2,3,1)), srf) # [B, H, W, 3]
    if clip_negative:
        rgb = torch.clamp(rgb.permute(0,3,1,2), min=0)

    return rgb # [B, 3, H, W]

def add_noise(signal, noise_type='poisson', npe=0, divFactorTo_1PE=1, seed=0):
    """
    Add noise to input signal, depending on the noise type.

    Args:
        signal: A tensor for the input signal of shape [B, C, H, W]. Default units: [Npe]
        noise_type: Type of noise. Could be either Poisson noise or Gaussian noise.
        npe: A parameter to control the noise level. 0 means no shot noise.
        divFactorTo_1PE: A factor to convert the input signal to Npe units. Default 1.
        seed: Random seed for the noise generation.

    Returns:
        noisy_signal: Noisy signal with the prescribed noise.
    """
    if npe == 0:
        return signal

    scale = npe / divFactorTo_1PE
    if noise_type == 'poisson':
        noisy_signal = torch.poisson(signal.clamp(min=0) * scale)
        noisy_signal = noisy_signal / scale
    elif noise_type == 'gaussian':
        gauss = torch.normal(torch.zeros(signal.shape).to(signal.device), scale)
        noisy_signal = signal + gauss
    else:
        raise ValueError(f"Unknown noise type {noise_type}.")
    
    return torch.clamp(noisy_signal, min=0.0)

def isp(image, config, bit=8):
    """
    Apply a simple ISP to the input image. Currently, only automatic exposure 
    is considered. Clipping of negative and max values are considered, however,
    quantization is not considered. 

    Args:
        image: A tensor for the input image signal.
        config: Configuration for the camera ISP.
        bit: Bit depth of the output image.

    Returns:
        image_isp: A digital image after applying the ISP.
    """
    if bit == 8:
        max_val = config.MAX_VAL_8_BIT
    elif bit == 12:
        max_val = config.MAX_VAL_12_BIT
    else:
        raise ValueError(f"Bit depth {bit} is not supported")
    # automatic exposure
    image_isp = image * (config.TYPICAL_SCENE_REFLECTIVITY / image.mean()) * max_val
    # quantization
    image_isp = image_isp.clamp(min=0, max=max_val)

    return image_isp

def crop_or_pad(in_tensor, shape=512):
    """
    Crop or pad the in_tensor in the last two dimensions to match the required shape.

    Args:
        in_tensor: An input tensor with shape [..., iM, iN].
        shape: The shape of the output shape. Should be a scalar or a tuple of two.

    Returns:
        out_tensor: Cropped or padded tensor.
    """
    
    if np.isscalar(shape):
        shape = (shape, shape)
    elif len(shape) > 2:
        raise ValueError(f"Shape should be a scalar or tuple of two, but got {len(shape)}.")
    
    refH, refW = shape
    inH, inW = in_tensor.shape[-2:]
    
    out_tensor = in_tensor.clone()
    
    # rows
    if inH > refH:
        # crop rows
        padH = (inH - refH) // 2
        out_tensor = out_tensor[..., padH:-(inH-refH-padH), :]
    elif inH < refH:
        # pad rows (top and bottom)
        padH = (refH - inH) // 2
        out_tensor = F.pad(out_tensor, (0, 0, padH, refH-inH-padH), 'constant', 0)
    # columns
    if inW > refW:
        # crop columns
        padW = (inW - refW) // 2
        out_tensor = out_tensor[..., padW:-(inW-refW-padW)]
    elif inW < refW:
        # pad columns (left and right)
        padW = (refW - inW) // 2
        out_tensor = F.pad(out_tensor, (padW, refW-inW-padW), 'constant', 0)
    
    return out_tensor

def conv2D(img, psf):
    """
    Apply 2D convolution between the input image tensor and psf tensor.

    Args:
        img: A tensor for the input image of shape [B, C, H, W].
        psf: A tensor for the PSF of shape [C, M, N].

    Returns:
        convolved: A tensor for the convolved image of shape [B, C, H, W].
    """
    # normalize the PSF in each channel
    psf = psf / torch.sum(psf, dim=(-2, -1), keepdim=True)
    # process in the Fourier domain 
    convolved = fft.ifftshift(fft.ifft2(fft.fft2(fft.fftshift(img)) * fft.fft2(fft.fftshift(psf)))).real
    
    return convolved

def image_formation(
    hsi, srf, config, psf=None, 
    noise_type='poisson', npe=0, divFactorTo_1PE=1, bit=8,
    seed=0,
):
    """
    Perform the image formation model to generate an RGB image from a hyperspectral datacube.

    Args:
        hsi: A tensor for the hyperspectral datacube with shape [B, C, H, W].
        srf: Spectral response function of the sensor with shape [C, 3].
        config: Configuration for the imaging process.
        psf: A numpy array for the spectral point spread function with shape [C, H, W].
        npe: A parameter to control the noise level. Defaul to 0 (no noise).
        divFactorTo_1PE: A factor to convert the input signal to Npe units. Default 1.
        bit: Bit depth of the RGB image.
        seed: Random seed for the noise generation.

    Returns:
        rgb: A numpy array for the RGB image, with shape [H, W, 3].
    """
    if psf is not None:
        # image formation by 2D convolution
        psf = crop_or_pad(psf, hsi.shape[-2:])
        # transpose before convolution
        _hsi = conv2D(hsi, psf)
    else:
        _hsi = hsi

    # 1. Project the hyperspectral datacube to RGB
    rgb = projectHSItoRGB(_hsi, srf, clip_negative=True)
    # 2. Add noise
    rgb = add_noise(rgb, noise_type=noise_type, npe=npe, divFactorTo_1PE=divFactorTo_1PE, seed=seed)
    # 3. Apply ISP
    rgb = isp(rgb, config, bit=bit)

    return rgb