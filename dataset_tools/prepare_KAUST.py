import sys
sys.path.append('..')
sys.path.append('.')

import os
import glob
import numpy as np
from scipy.io import loadmat
import argparse

import metamer.utils as utils
from config import kaust as kaust_config

def make_kaust(args):
    """
    Prepare the KAUST dataset.

    The dataset consists of 409 scenes. Spatial resolution is 512 x 512. 
    Spectral range is 0.40um - 0.73um at 0.01um steps (34 bands). 
    Each datacube is stored as a .mat file.

    We normalize the data by the maximum possible value of 12 bit (4095). 
    The save mat file is [31, 512w, 512h].
    """
    print("convert mat files for KAUST")
    if not os.path.exists(f"{args.dest_root}/Train_Spec"):
        os.mkdir(f"{args.dest_root}/Train_Spec")
    
    files = sorted(glob.glob(f'{args.dataset_root}/*.mat'))
    hyper_list = []
    for i in range(len(files)):
        fn = os.path.split(files[i])[-1][:-4]
        print(f"{i}: {fn}")
        hyper_list.append(fn)
        mat = loadmat(files[i])
        hsi = np.array(mat['hsi']) / 4095 # [512w, 512h, 34]
        hsi = hsi[:,:,:31] # [512w, 512h, 31]
        hsi = hsi.transpose(1,0,2) # [512h, 512w, 31]
        # saved mat will be in reversed order [31, 512w, 512h]
        utils.save_hsi_mat(f"{args.dest_root}/Train_Spec/{fn}.mat", hsi, bands=kaust_config.WAVELENGTHS)
    # split txt
    print("split train set and validation set")
    split_dir = f"{args.dest_root}/split_txt"
    if not os.path.exists(split_dir):
        # 90% for training (368), and 10% for validation (41)
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
    Prepare the KAUST dataset.
    """

    parser = argparse.ArgumentParser(
        description='Prepare KAUST dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--dataset_root', type=str, default='/path/to/original/KAUST', help='path to original dataset')
    parser.add_argument('--dest_root', type=str, default='./datasets/KAUST', help='destination path')

    args = parser.parse_args()

    make_kaust(args)

