import sys
sys.path.append('..')
sys.path.append('.')

import os
import glob
import numpy as np
from PIL import Image
import argparse

from metamer.utils import save_hsi_mat
from config import cave as cave_config

ignore_list = ['watercolors_ms']

def make_cave(args):
    """
    Prepare the CAVE dataset.

    The dataset consists of 32 scenes. Spatial resolution is 512 x 512.
    Spectral range is 0.4um - 0.7um at 0.01um steps (31 bands). Each band is stored
    as a 16-bit grayscale PNG image. Image filenames are of the format 'object_ms_01.png',
    where the '01' at the end signifies that this is the first image (captured at 0.4um). 
    Thus, '02' corresponds to 0.41um, and so on, until '31' for 0.7um.

    We ignore 'watercolors_ms', which is problematic.

    The data is saved in matlab compatible mat files, with shape [31, 512w, 512h].

    Dataset URL: https://cave.cs.columbia.edu/repository/Multispectral
    
    Download link: https://cave.cs.columbia.edu/old/databases/multispectral/zip/complete_ms_data.zip
    """
    # convert png to mat
    print("convert png to mat files for CAVE")
    if not os.path.exists(f"{args.dest_root}/Train_Spec"):
        os.mkdir(f"{args.dest_root}/Train_Spec")
    tree = list(os.walk(args.dataset_root))
    hyper_list = []
    for i in range(len(tree[0][1])):
        fn = tree[0][1][i]
        print(f"{i}: {fn}")
        if fn in ignore_list:
            continue
        hyper_list.append(fn)
        files = sorted(glob.glob(f'{args.dataset_root}/{fn}/{fn}/{fn}_*.png'))
        hsi_list = []
        for path in files:
            # raw data is 16-bit, hsi is normalized by the largest possible value
            image = np.array(Image.open(path)) / (2**16-1)
            hsi_list.append(image)
        hsi = np.array(hsi_list).astype(np.float32) # [31, 512h, 512w]
        hsi = hsi.transpose(1, 2, 0) # [512h, 512w, 31], important, keep consistent apperance
        # saved mat will be in reversed order [31, 512w, 512h]
        save_hsi_mat(f"{args.dest_root}/Train_Spec/{fn}.mat", hsi, bands=cave_config.WAVELENGTHS)
    # split txt
    print("split train set and validation set")
    split_dir = f"{args.dest_root}/split_txt"
    if not os.path.exists(split_dir):
        os.mkdir(split_dir)
        print("No existing split found, generating default split files...")
        # 90% for training (27), and 10% for validation (4)
        print(f"total number of hsi: {len(hyper_list)}")
        split = int(np.floor(0.9 * len(hyper_list)))
        print(f"number of train set: {split}")
        print(f"number of validation set: {len(hyper_list) - split}")
        with open(f"{split_dir}/train_list.txt", 'w') as f:
            for filename in hyper_list[:split]:
                f.write(filename + '\n')
        with open(f"{split_dir}/valid_list.txt", 'w') as f:
            for filename in hyper_list[split:]:
                f.write(filename + '\n')
    else:
        # To ensure reproducibility, please use the provided train/valid lists.
        print("Existing split found, skipping generation of train/valid lists.")

if __name__ == '__main__':
    """
    Prepare the CAVE dataset.
    """

    parser = argparse.ArgumentParser(
        description='Prepare CAVE dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--dataset_root', type=str, default='/path/to/original/CAVE', help='path to original dataset')
    parser.add_argument('--dest_root', type=str, default='./datasets/CAVE', help='destination path')

    args = parser.parse_args()

    make_cave(args)

