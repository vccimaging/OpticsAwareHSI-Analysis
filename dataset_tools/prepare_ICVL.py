import sys
sys.path.append('..')
sys.path.append('.')

import os
import glob
import numpy as np
import h5py
import argparse

import metamer.utils as utils
from config import icvl as icvl_config

def make_icvl(args):
    """
    Prepare the ICVL dataset.

    The original dataset consists of 201 scenes. Spatial resolution is 1392 x 1300. 
    Spectral range is 0.40um - 0.70um at 0.1um steps (31 bands). Each datacube 
    is stored as a .mat file.

    We normalize the data by the maximum possible value of 12 bit (4095), and rotate 
    the spatial dimension by 90 degrees so that the scene is up right. The save mat file
    is [31, 1300, 1392].

    Dataset URL: https://icvl.cs.bgu.ac.il/hyperspectral/
    """
    print("convert mat files for ICVL")
    if not os.path.exists(f"{args.dest_root}/Train_Spec"):
        os.mkdir(f"{args.dest_root}/Train_Spec")
    
    files = sorted(glob.glob(f'{args.dataset_root}/*.mat'))
    hyper_list = []
    for i in range(len(files)):
        fn = os.path.split(files[i])[-1][:-4]
        print(f"{i}: {fn}")
        hyper_list.append(fn)
        mat = h5py.File(files[i], 'r')
        hsi = np.array(mat['rad']) / 4095 # [31, 1392, 1300]
        hsi = hsi.astype(np.float32)
        hsi = hsi.T # [1300, 1392, 31]
        hsi = np.rot90(hsi, k=1) # rotate 90 deg couterclockwise, [1392, 1300, 31]
        # saved mat will be [31, 1300, 1392]
        utils.save_hsi_mat(f"{args.dest_root}/Train_Spec/{fn}.mat", hsi, bands=icvl_config.WAVELENGTHS)
    # split txt
    print("split train set and validation set")
    split_dir = f"{args.dest_root}/split_txt"
    if not os.path.exists(split_dir):
        # 90% for training (180), and 10% for validation (21)
        print(f"total number of hsi: {len(hyper_list)}")
        split = int(np.floor(0.9*len(hyper_list)))
        print(f"number of train set: {split}")
        print(f"number of validation set: {len(hyper_list) - split}")
        with open(f"{args.dest_root}/split_txt/train_list.txt", 'w') as f:
            for filename in hyper_list[:split]:
                f.write(filename + '\n')
        with open(f"{args.dest_root}/split_txt/valid_list.txt", 'w') as f:
            for filename in hyper_list[split:]:
                f.write(filename + '\n')
    else:
        # To ensure reproducibility, please use the provided train/valid lists.
        print("Existing split found, skipping generation of train/valid lists.")

if __name__ == '__main__':
    """
    Prepare the ICVL dataset.
    """

    parser = argparse.ArgumentParser(
        description='Prepare ICVL dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--dataset_root', type=str, default='/path/to/original/ICVL', help='path to original dataset')
    parser.add_argument('--dest_root', type=str, default='./datasets/ICVL', help='destination path')

    args = parser.parse_args()

    make_icvl(args)

