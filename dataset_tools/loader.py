"""
This code is heavily borrowed MST++ with necessary corrections and enrichment.

The original code can be found at
https://github.com/caiyuanhao1998/MST-plus-plus/blob/master/test_develop_code/hsi_dataset.py
Accessed on: 2025-04-23
"""

import numpy as np
import random
import cv2
import h5py
from torch.utils.data import Dataset
import metamer.utils as utils
import torch
from skimage import segmentation

class TrainDataset(Dataset):
    def __init__(
        self, 
        original_data_root, 
        metamer_data_root, 
        crop_size=128, 
        stride=8, 
        data_join="both", 
        bgr2rgb=True, 
        rgb_format='png', 
        percent=None, 
        aug=True, 
        on_the_fly=False, 
        args=None, 
        config=None,
        method=None
    ):
        """
        Create the training dataset.

        Args:
            original_data_root: Root path to the original hsi data.
            metamer_data_root: Root path to the metamer hsi data.
            crop_size: Size of the cropped patch.
            stride: An integer for the stride to jump from one pixel to another.
            data_join: Method for preparing the validation data. Options are 'standard', 'metamer', and 'both'.
                'standard' means the original data.
                'metamer' means the generated metameric data with fixed metameric black coefficient alpha = 0.
                'both' means both the original data and the generated metamer data.
            bgr2rgb: Whether to convert from BGR to RGB (True) or not (False).
            rgb_format: Format of the RGB images. Default to lossless png.
            percent: Percentage (10 - 100) of training data to use. Default is full (None).
            aug: Whether to spatially augment the data or not.
            on_the_fly: Whether to create metamer on the fly (True) or not (False).
            args: Argumnets that contain necessary information for the SRF, PSF, etc.
            config: Configuration for the image formation. Used only for on-the-fly metamer situation.
            method: The evaluated method. Only used to deal special case of HPRN.
        """

        self.hypers = []
        self.bgrs = []
        self.method = method
        self.crop_size = crop_size
        self.stride = stride
        h, w = config.HEIGHT, config.WIDTH
        self.patch_per_line = (w-crop_size)//stride+1
        self.patch_per_colum = (h-crop_size)//stride+1
        self.patch_per_img = self.patch_per_line*self.patch_per_colum
        self.aug = aug
        self.on_the_fly = on_the_fly
        self.args = args
        self.config = config

        if self.on_the_fly:
            # In on-the-fly case, metamers are generated dynamically using the image formation model
            self.SRF = utils.load_srf(self.args.srf_path, dtype=np.float32)
            self.SRF = torch.from_numpy(self.SRF)
            try:
                self.PSFs = np.load(f"{args.psf_path}/{args.psf_name}.npz")['PSFs'].astype(np.float32)
                self.PSFs = torch.from_numpy(self.PSFs.transpose(2,0,1)) # [H, W, C] -> [C, H, W]
            except:
                if (str(args.psf_name).lower() == 'none') or (args.psf_name is None):
                    self.PSFs = None
                else:
                    raise ValueError(f"No such file {args.psf_path} for PSF.")

        # standard hsi data path
        hyper_data_path = f'{original_data_root}/Train_Spec/'
        # metamer hsi data path
        hyper_data_path_metamer = f'{metamer_data_root}/Train_Metamer_Spec/'
        # color image path for both standard and metamer data
        bgr_data_path = f'{metamer_data_root}/Train_Metamer_RGB/'

        # data lists
        with open(f'{original_data_root}/split_txt/train_list.txt', 'r') as fin:
            if data_join == "both": 
                # hyper list for the metamer part
                hyper_list_metamer = [line.replace('\n', '_metamer.mat') for line in fin]
                # bgr list for the metamer part
                if rgb_format == 'png':
                    bgr_list_metamer = [line.replace('.mat','.png') for line in hyper_list_metamer]
                elif rgb_format == 'jpg':
                    bgr_list_metamer = [line.replace('.mat','.jpg') for line in hyper_list_metamer]
                else:
                    raise ValueError(f"Unknown image format {rgb_format}")

                # hyper list for the standard part
                hyper_list_std = [line.replace('_metamer','') for line in hyper_list_metamer]
                # bgr list for the standard part
                if rgb_format == 'png':
                    bgr_list_std = [line.replace('.mat','_original.png') for line in hyper_list_std]
                elif rgb_format == 'jpg':
                    bgr_list_std = [line.replace('.mat','_original.jpg') for line in hyper_list_std]
                else:
                    raise ValueError(f"Unknown image format {rgb_format}")
                
                if percent is not None:
                    subset = int(np.floor(len(hyper_list) * percent / 100))
                    if subset < len(hyper_list):
                        start = np.random.randint(0, len(hyper_list)-subset)
                        hyper_list_metamer = hyper_list_metamer[start:start+subset+1]
                        hyper_list_std = hyper_list_std[start:start+subset+1]
                        bgr_list_metamer = bgr_list_metamer[start:start+subset+1]
                        bgr_list_std = bgr_list_std[start:start+subset+1]
                
                # concatenate the two parts
                hyper_list = hyper_list_metamer + hyper_list_std
                bgr_list = bgr_list_metamer + bgr_list_std

            elif data_join == "standard":
                # hyper list for the standard part
                hyper_list = [line.replace('\n', '.mat') for line in fin]
                # bgr list for the standard part
                if rgb_format == 'png':
                    bgr_list = [line.replace('.mat','_original.png') for line in hyper_list]
                elif rgb_format == 'jpg':
                    bgr_list = [line.replace('.mat','_original.jpg') for line in hyper_list]
                else:
                    raise ValueError(f"Unknown image format {rgb_format}")
                
                if percent is not None:
                    subset = int(np.floor(len(hyper_list) * percent / 100))
                    if subset < len(hyper_list):
                        start = np.random.randint(0, len(hyper_list)-subset)
                        hyper_list = hyper_list[start:start+subset+1]
                        bgr_list = bgr_list[start:start+subset+1]
            
            elif data_join == "metamer":
                # hyper list for the metamer part
                hyper_list = [line.replace('\n', '_metamer.mat') for line in fin]
                # bgr list for the metamer part
                if rgb_format == 'png':
                    bgr_list = [line.replace('.mat','.png') for line in hyper_list]
                elif rgb_format == 'jpg':
                    bgr_list = [line.replace('.mat','.jpg') for line in hyper_list]
                else:
                    raise ValueError(f"Unknown image format {rgb_format}")
                
                if percent is not None:
                    subset = int(np.floor(len(hyper_list) * percent / 100))
                    if subset < len(hyper_list):
                        start = np.random.randint(0, len(hyper_list)-subset)
                        hyper_list = hyper_list[start:start+subset+1]
                        bgr_list = bgr_list[start:start+subset+1]
                
            else:
                raise ValueError(f"Unknown data join {data_join}")
                
        indices = list(range(len(bgr_list)))
        random.shuffle(indices)

        hyper_list = [hyper_list[i] for i in indices]
        bgr_list = [bgr_list[i] for i in indices]
        print(f'len(hyper_train) of dataset:{len(hyper_list)}')
        print(f'len(bgr_train) of dataset:{len(bgr_list)}')

        # data
        for i in range(len(hyper_list)):
            if "metamer" in hyper_list[i]:
                hyper_path = hyper_data_path_metamer + hyper_list[i]
            else: 
                hyper_path = hyper_data_path + hyper_list[i]

            if 'mat' not in hyper_path:
                continue
            with h5py.File(hyper_path, 'r') as mat:
                hyper = np.float32(np.array(mat['cube'])) # e.g., [31, 512, 482]
            hyper = np.transpose(hyper, [0, 2, 1]) # e.g., [31, 482, 512]

            bgr_path = bgr_data_path + bgr_list[i]
            bgr = cv2.imread(bgr_path)
            if bgr2rgb:
                bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            bgr = np.float32(bgr)
            
            try:
                assert bgr.shape == (h,w,3)
                assert hyper.shape == (31,h,w)

                bgr = (bgr - bgr.min()) / (bgr.max() - bgr.min())
                bgr = np.transpose(bgr, [2, 0, 1]) # e.g., [3, 482, 512]
                self.hypers.append(hyper)
                self.bgrs.append(bgr)
                mat.close()
                print(f'Scene {i}: {hyper_list[i]} and {bgr_list[i]} are loaded.')

            except:
                print("BGR: ",bgr.shape,"Hyper: ", hyper.shape)
                print(f'Ignore {i}: {hyper_list[i]} and {bgr_list[i]}.')

        # data info
        self.img_num = len(self.hypers)
        self.length = self.patch_per_img * self.img_num

    def augment(self, img, rotTimes, vFlip, hFlip):
        """
        Spatially augment the data.

        Args:
            img: Original input image data.
            rotTimes: Number of rotation times.
            vFlip: Number of vertical flips.
            hFlip: Number of horizontal flips.
        """
        # Random rotation
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(1, 2))
        # Random vertical Flip
        for j in range(vFlip):
            img = img[:, :, ::-1].copy()
        # Random horizontal Flip
        for j in range(hFlip):
            img = img[:, ::-1, :].copy()
        return img

    def __getitem__(self, idx):
        stride = self.stride
        crop_size = self.crop_size
        img_idx, patch_idx = idx//self.patch_per_img, idx%self.patch_per_img
        h_idx, w_idx = patch_idx//self.patch_per_line, patch_idx%self.patch_per_line
        bgr = self.bgrs[img_idx]
        hyper = self.hypers[img_idx]
        bgr = bgr[:,h_idx*stride:h_idx*stride+crop_size, w_idx*stride:w_idx*stride+crop_size]
        hyper = hyper[:, h_idx * stride:h_idx * stride + crop_size,w_idx * stride:w_idx * stride + crop_size]
        
        # spatial augmentation options
        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)
        if self.aug:
            bgr = self.augment(bgr, rotTimes, vFlip, hFlip)
            hyper = self.augment(hyper, rotTimes, vFlip, hFlip)

        if self.on_the_fly:
            # generate metamer data on-the-fly
            alpha = np.random.uniform(-1.0, 2.0, 1).item() # random coefficient for metameric black
            meta_aug = utils.MetamerAugment(self.SRF, alpha)
            hsi_metamer = meta_aug(torch.from_numpy(np.ascontiguousarray(hyper)))

            # generate RGB data with image formation model
            images = utils.image_formation(
                hsi_metamer, self.SRF, self.config, self.PSFs, 
                noise_type=self.args.noise_type, npe=self.args.npe, 
                seed=self.args.seed
            )

            images = (images - images.min()) / (images.max() - images.min())

            images = images.squeeze(0)
            hsi_metamer = hsi_metamer.squeeze(0)

            if self.method == 'hprn':
                semantic_label_list = []
                slic_scales = [8, 12, 16, 20]
                slic_input = np.uint8(images * 255.0)
                slic_input = np.transpose(slic_input, (1, 2, 0))  # [C, H, W] -> [H, W, C]
                for s in slic_scales:
                    semantic_label_list.append(segmentation.slic(slic_input, start_label=1, n_segments=s)[None, :])
                semantic_labels = np.concatenate(semantic_label_list, axis=0)
                return [np.ascontiguousarray(images), np.ascontiguousarray(semantic_labels)], np.ascontiguousarray(hsi_metamer)
            
            else:
                return images, hsi_metamer

        else:
            if self.method == 'hprn':
                # HRPN requires semantic lables
                semantic_label_list = []
                slic_scales = [8, 12, 16, 20]
                slic_input = np.uint8(bgr * 255.0)
                slic_input = np.transpose(slic_input, (1, 2, 0))  # [C, H, W] -> [H, W, C]
                for s in slic_scales:
                    semantic_label_list.append(segmentation.slic(slic_input, start_label=1, n_segments=s)[None, :])
                semantic_labels = np.concatenate(semantic_label_list, axis=0)

                return [np.ascontiguousarray(bgr), np.ascontiguousarray(semantic_labels)], np.ascontiguousarray(hyper)
            
            else:
                return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)

    def __len__(self):
        return self.patch_per_img*self.img_num

class ValidDataset(Dataset):
    def __init__(
        self, 
        original_data_root, 
        metamer_data_root, 
        data_join="both", 
        bgr2rgb=True, 
        rgb_format='png', 
        method=None,
    ):
        """
        Create the validation dataset.

        Args:
            original_data_root: Root path to the original hsi data.
            metamer_data_root: Root path to the metamer hsi data.
            data_join: Method for preparing the validation data. Options are 'standard', 'metamer', and 'both'.
                'standard' means the original data.
                'metamer' means the generated metameric data with fixed metameric black coefficient alpha = 0.
                'both' means both the original data and the generated metamer data.
            bgr2rgb: Whether to convert from BGR to RGB (True) or not (False).
            rgb_format: Format of the RGB images. Default to lossless png.
            method: The evaluated method. Only used to deal special case of HPRN.
        """
        self.hypers = []
        self.bgrs = []
        self.method = method

        # data paths
        hyper_data_path_metamer = f'{metamer_data_root}/Train_Metamer_Spec/'
        hyper_data_path = f'{original_data_root}/Train_Spec/'
        bgr_data_path = f'{metamer_data_root}/Train_Metamer_RGB/'

        # data lists
        with open(f'{original_data_root}/split_txt/valid_list.txt', 'r') as fin:
            if data_join == "both": 
                # hyper list for the metamer part
                hyper_list_metamer = [line.replace('\n', '_metamer.mat') for line in fin]
                # bgr list for the metamer part
                if rgb_format == 'png':
                    bgr_list_metamer = [line.replace('mat','png') for line in hyper_list_metamer]
                elif rgb_format == 'jpg':
                    bgr_list_metamer = [line.replace('mat','jpg') for line in hyper_list_metamer]
                else:
                    raise ValueError(f"Unknown image format {rgb_format}")

                # hyper list for the standard part
                hyper_list_std = [line.replace('_metamer','') for line in hyper_list_metamer]
                # bgr list for the standard part
                if rgb_format == 'png':
                    bgr_list_std = [line.replace('.mat','_original.png') for line in hyper_list_std]
                elif rgb_format == 'jpg':
                    bgr_list_std = [line.replace('.mat','_original.jpg') for line in hyper_list_std]
                else:
                    raise ValueError(f"Unknown image format {rgb_format}")
                
                # concatenate the two parts
                hyper_list = hyper_list_metamer + hyper_list_std
                bgr_list = bgr_list_metamer + bgr_list_std

            elif data_join == "standard":
                # hyper list for the standard part
                hyper_list = [line.replace('\n', '.mat') for line in fin]
                # bgr list for the standard part
                if rgb_format == 'png':
                    bgr_list = [line.replace('.mat','_original.png') for line in hyper_list]
                elif rgb_format == 'jpg':
                    bgr_list = [line.replace('.mat','_original.jpg') for line in hyper_list]
                else:
                    raise ValueError(f"Unknown image format {rgb_format}")
            
            elif data_join == "metamer":
                # hyper list for the metamer part
                hyper_list = [line.replace('\n', '_metamer.mat') for line in fin]
                # bgr list for the metamer part
                if rgb_format == 'png':
                    bgr_list = [line.replace('mat','png') for line in hyper_list]
                elif rgb_format == 'jpg':
                    bgr_list = [line.replace('mat','jpg') for line in hyper_list]
                else:
                    raise ValueError(f"Unknown image format {rgb_format}")
                
            else:
                raise ValueError(f"Unknown data join {data_join}")
            
        print(f'len(hyper_valid) of dataset:{len(hyper_list)}')
        print(f'len(bgr_valid) of dataset:{len(bgr_list)}')

        # data
        for i in range(len(hyper_list)):
            if "metamer" in hyper_list[i]:
                hyper_path = hyper_data_path_metamer + hyper_list[i]
            else: 
                hyper_path = hyper_data_path + hyper_list[i]

            if 'mat' not in hyper_path:
                continue
            with h5py.File(hyper_path, 'r') as mat:
                hyper = np.float32(np.array(mat['cube']))
            hyper = np.transpose(hyper, [0, 2, 1])
            bgr_path = bgr_data_path + bgr_list[i]
            
            bgr = cv2.imread(bgr_path)
            if bgr2rgb:
                bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            bgr = np.float32(bgr)
            bgr = (bgr - bgr.min()) / (bgr.max() - bgr.min())
            bgr = np.transpose(bgr, [2, 0, 1])
            self.hypers.append(hyper)
            self.bgrs.append(bgr)
            mat.close()
            print(f'Scene {i}: {hyper_list[i]} and {bgr_list[i]} are loaded.')

    def __getitem__(self, idx):
        hyper = self.hypers[idx]
        bgr = self.bgrs[idx]
        if self.method == 'hprn':
            # HPRN requires the semantic labels
            semantic_label_list = []
            slic_scales = [8, 12, 16, 20]
            slic_input = np.uint8(bgr * 255.0)
            slic_input = np.transpose(slic_input, (1, 2, 0))  # [C, H, W] -> [H, W, C]
            for s in slic_scales:
                semantic_label_list.append(segmentation.slic(slic_input, start_label=1, n_segments=s)[None, :])
            semantic_labels = np.concatenate(semantic_label_list, axis=0)
            return [np.ascontiguousarray(bgr), np.ascontiguousarray(semantic_labels)], np.ascontiguousarray(hyper)
        else:
            return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)

    def __len__(self):
        return len(self.hypers)