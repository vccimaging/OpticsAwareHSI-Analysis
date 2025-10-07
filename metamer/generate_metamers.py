import os
import sys
sys.path.append('..')
sys.path.append('.')

import shutil
import argparse
from importlib.machinery import SourceFileLoader
import numpy as np
import cv2
import utils

def create_metamer_data(args, config, data_list):
    """
    Create metamer data from existing dataset.
    """
    # load SRF and PSF data
    SRF = utils.load_srf(args.srf_path)
    try:
        PSFs = np.load(f"{args.psf_path}/{args.psf_name}.npz")['PSFs']
    except:
        if (str(args.psf_name).lower() == 'none') or (args.psf_name is None):
            PSFs = None
        else:
            raise ValueError(f"No such file {args.psf_path} for PSF.")

    # create metamers
    for (i, fn) in enumerate(data_list):
        print(f"{i}: {fn}")
        # original hsi, [H, W, C]
        HSI = utils.load_hsi_mat(args.original_hsi_path+fn+'.mat', name='cube', transpose=True)
        # metamer augmented hsi, [H, W, C]
        HSI_metamer = utils.generate_metamer_np(HSI, SRF, alpha=args.alpha)
        HSI_metamer = HSI_metamer.astype(HSI.dtype) # save in the same dtype
        # save augmented hsi, [C, W, H]
        utils.save_hsi_mat(args.metamer_hsi_path+fn+'_metamer.mat', HSI_metamer, bands=config.WAVELENGTHS)
        # image formation to generate RGB images
        rgb = utils.image_formation_np(
            HSI, SRF, config, psf=PSFs, 
            noise_type=args.noise_type, npe=args.npe, divFactorTo_1PE=args.divFactorTo_1PE, seed=args.noise_seed
        )
        rgb_metamer = utils.image_formation_np(
            HSI_metamer, SRF, config, psf=PSFs, 
            noise_type=args.noise_type, npe=args.npe, divFactorTo_1PE=args.divFactorTo_1PE, seed=args.noise_seed
        )
        # save original and augmented rgb images
        if args.rgb_format == 'png':
            # save in lossless png
            cv2.imwrite(
                args.metamer_rgb_path+fn+'_original.png', 
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), 
                [cv2.IMWRITE_PNG_COMPRESSION, 0]
            )
            cv2.imwrite(
                args.metamer_rgb_path+fn+'_metamer.png', 
                cv2.cvtColor(rgb_metamer, cv2.COLOR_RGB2BGR), 
                [cv2.IMWRITE_PNG_COMPRESSION, 0]
            )
        elif args.rgb_format == 'jpg':
            # save in compressed JPEG 
            cv2.imwrite(
                args.metamer_rgb_path+fn+'_original.jpg', 
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), 
                [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY]
            )
            cv2.imwrite(
                args.metamer_rgb_path+fn+'_metamer.jpg', 
                cv2.cvtColor(rgb_metamer, cv2.COLOR_RGB2BGR), 
                [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY]
            )
        elif args.rgb_format == 'all':
            # save in both lossless png and compressed JPEG 
            cv2.imwrite(
                args.metamer_rgb_path+fn+'_original.png', 
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), 
                [cv2.IMWRITE_PNG_COMPRESSION, 0]
            )
            cv2.imwrite(
                args.metamer_rgb_path+fn+'_metamer.png', 
                cv2.cvtColor(rgb_metamer, cv2.COLOR_RGB2BGR), 
                [cv2.IMWRITE_PNG_COMPRESSION, 0]
            )
            cv2.imwrite(
                args.metamer_rgb_path+fn+'_original.jpg', 
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), 
                [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY]
            )
            cv2.imwrite(
                args.metamer_rgb_path+fn+'_metamer.jpg', 
                cv2.cvtColor(rgb_metamer, cv2.COLOR_RGB2BGR), 
                [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY]
            )
        else:
            raise ValueError(f"Unknown RGB format {args.rgb_format}")

if __name__ == "__main__":
    """
    Generate metamers from existing datasets.
    """

    parser = argparse.ArgumentParser(
        description='Metamer generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--srf_path', type=str, default='./resources/A2a5320-23ucBAS_gain.npz',help='path to spectral response function')
    parser.add_argument('--psf_path', type=str, default='./PSFs', help='path to point spread function')
    parser.add_argument('--psf_name', type=str, default=None, help='point spread function name')
    parser.add_argument('--dataset_path', type=str, default='./datasets', help='path to datasets')
    parser.add_argument('--dataset_name', type=str, default='ARAD_1K', help='dataset name')

    parser.add_argument('--noise_type', type=str, default='poisson', choices=['poisson', 'gaussian'], help='noise type')
    parser.add_argument('--npe', type=int, default=0, help='a parameter to control the noise level')
    parser.add_argument('--divFactorTo_1PE', type=int, default=1, help='a factor to convert the signal to Npe units')
    parser.add_argument('--noise_seed', type=int, default=0, help='random seed for the noise')
    parser.add_argument('--alpha', type=float, default=0.0, help='coefficient for the metameric black')
    parser.add_argument('--rgb_format', type=str, default='png', choices=['png', 'jpg', 'all'], help='format for the RGB image')

    args = parser.parse_args()

    if args.dataset_name.lower() == 'arad_1k':
        config = SourceFileLoader('config', './config/arad_1k.py').load_module()
    elif args.dataset_name.lower() == 'cave':
        config = SourceFileLoader('config', './config/cave.py').load_module()
    elif args.dataset_name.lower() == 'icvl':
        config = SourceFileLoader('config', './config/icvl.py').load_module()
    elif args.dataset_name.lower() == 'kaust':
        config = SourceFileLoader('config', './config/kaust.py').load_module()
    else:
        raise ValueError(f"Unknown dataset {args.dataset_name}.")

    # data folders
    args.original_hsi_path = f'{args.dataset_path}/{args.dataset_name}/Train_Spec/'
    args.metamer_hsi_path = f'{args.dataset_path}/{args.dataset_name}/{args.noise_type}{args.npe}_{args.psf_name}/Train_Metamer_Spec/'
    args.metamer_rgb_path = f'{args.dataset_path}/{args.dataset_name}/{args.noise_type}{args.npe}_{args.psf_name}/Train_Metamer_RGB/'

    # main folder
    folder = f"{args.dataset_path}/{args.dataset_name}/{args.noise_type}{args.npe}_{args.psf_name}"
    if not os.path.exists(folder):
        os.mkdir(folder)
    else:
        shutil.rmtree(folder)
        os.mkdir(folder)
    # metamer HSI folder
    if not os.path.exists(args.metamer_hsi_path):
        os.mkdir(args.metamer_hsi_path)
    else:
        shutil.rmtree(args.metamer_hsi_path)
        os.mkdir(args.metamer_hsi_path)
    # metamer RGB folder
    if not os.path.exists(args.metamer_rgb_path):
        os.mkdir(args.metamer_rgb_path)
    else:
        shutil.rmtree(args.metamer_rgb_path)
        os.mkdir(args.metamer_rgb_path)
    # training data list
    with open(f'{args.dataset_path}/{args.dataset_name}/split_txt/train_list.txt', 'r') as fin:
        train_list = [line.replace('\n', '') for line in fin]
    train_list.sort()
    # validation data list
    with open(f'{args.dataset_path}/{args.dataset_name}/split_txt/valid_list.txt', 'r') as fin:
        val_list = [line.replace('\n', '') for line in fin]
    val_list.sort()

    print(f"Creating metamer data for {args.dataset_name}, {args.noise_type} noise={args.npe}, psf={args.psf_name}")

    print("   training data ...")
    create_metamer_data(args, config, train_list)
    print("   validation data ...")
    create_metamer_data(args, config, val_list)
    print("Done!")
