import os
import sys
sys.path.append('..')
sys.path.append('.')

import argparse
from importlib.machinery import SourceFileLoader

import numpy as np
import pandas as pd

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
cudnn.benchmark = True

import metamer.utils as utils
import metamer.metrics as metrics
from metamer.utils import str2bool, load_srf

from architecture import *

# wavelengths
lambda1 = 0.4
lambda2 = 0.7
dlambda = 0.01
wavelengths = np.arange(lambda1, lambda2+dlambda, dlambda)

def validate(val_loader, model, args, val_list, config):
    """
    Validate pretrained model on validation data.

    Args:
        val_loader: Dataloader for the validation data.
        model: Pretrained model.
        args: Options.
        val_list: A list of filenames for the validation list.

    Returns:
        mrae: Average MRAE.
        rmse: Average RMSE.
        psnr: Average PSNR.
        sam: Average SAM.
    """
    model.eval()

    # loss functions
    criterion_mrae = metrics.Loss_MRAE()
    criterion_rmse = metrics.Loss_RMSE()
    criterion_psnr = metrics.Loss_PSNR()
    criterion_sam = metrics.Loss_SAM()
    if torch.cuda.is_available():
        criterion_mrae.cuda()
        criterion_rmse.cuda()
        criterion_psnr.cuda()
        criterion_sam.cuda()

    losses_mrae = metrics.AverageMeter()
    losses_rmse = metrics.AverageMeter()
    losses_psnr = metrics.AverageMeter()
    losses_sam = metrics.AverageMeter()
    
    for i, (input, target) in enumerate(val_loader):
        if torch.cuda.is_available():
                target = target.cuda()
        with torch.no_grad():
            if args.dataset_name == 'ICVL':
                # For ICVL, we crop the central 512 x 512 region for validation, to be consistent with others
                print("ICVL: center 512 x 512 region")
                M = config.HEIGHT // 2
                N = config.WIDTH // 2
                
                if args.method == 'hprn':
                    semantic_labels = (input[1][:, :, M-256:M+256, N-256:N+256]).cuda()
                    input = input[0].cuda()
                    output = model(input[:, :, M-256:M+256, N-256:N+256], semantic_labels.to(input.device))
                elif args.method == 'ssrnet':
                    input = input.cuda()
                    RGB_FILTER_CSV = './resources/A2a5320-23ucBAS_gain.npz' # SSRNet requires the SRF
                    SRF = load_srf(RGB_FILTER_CSV, dtype=np.float32)
                    SRF = torch.from_numpy(SRF).permute(1, 0)
                    SRF_normalized = (SRF/SRF.sum(-1, keepdims=True))
                    output = model(SRF_normalized.to(input.device), input[:, :, M-256:M+256, N-256:N+256])
                else:
                    input = input.cuda()
                    output = model(input[:, :, M-256:M+256, N-256:N+256])
                
                loss_mrae = criterion_mrae(output, target[:, :, M-256:M+256, N-256:N+256])
                loss_rmse = criterion_rmse(output, target[:, :, M-256:M+256, N-256:N+256])
                loss_psnr = criterion_psnr(output, target[:, :, M-256:M+256, N-256:N+256])
                loss_sam = criterion_sam(output, target[:, :, M-256:M+256, N-256:N+256])
            else:
                if args.method == 'hprn':
                    semantic_labels = input[1].cuda()
                    input = input[0].cuda()
                    output = model(input, semantic_labels.to(input.device))
                elif args.method == 'ssrnet':
                    input = input.cuda()
                    RGB_FILTER_CSV = './resources/A2a5320-23ucBAS_gain.npz'
                    SRF = load_srf(RGB_FILTER_CSV, dtype=np.float32)
                    SRF = torch.from_numpy(SRF).permute(1, 0)
                    SRF_normalized = (SRF/SRF.sum(-1, keepdims=True))
                    output = model(SRF_normalized.to(input.device), input)
                else:
                    input = input.cuda()
                    output = model(input)

                loss_mrae = criterion_mrae(output, target)
                loss_rmse = criterion_rmse(output, target)
                loss_psnr = criterion_psnr(output, target)
                loss_sam = criterion_sam(output, target)

        # record loss
        losses_mrae.update(loss_mrae.data)
        losses_rmse.update(loss_rmse.data)
        losses_psnr.update(loss_psnr.data)
        losses_sam.update(loss_sam.data)

        result = output.cpu().numpy() * 1.0
        result = np.transpose(np.squeeze(result), [1, 2, 0])
        result = np.minimum(result, 1.0)
        result = np.maximum(result, 0)
        if args.metamer:
            mat_name = val_list[i] + '_metamer.mat'
        else:
            mat_name = val_list[i] + '_std.mat'
        mat_dir = os.path.join(args.outf, mat_name)
        print(mat_name)
        utils.save_hsi_mat(mat_dir, result, bands=wavelengths)
    return losses_mrae.avg, losses_rmse.avg, losses_psnr.avg, losses_sam.avg

if __name__ == "__main__":
    """
    Validate pretrained models.
    """

    parser = argparse.ArgumentParser(
        description='RGB2HS model validation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--psf_name', type=str, default=None, help='point spread function name')
    parser.add_argument('--dataset_path', type=str, default='./datasets', help='path to datasets')
    parser.add_argument('--dataset_name', type=str, default='ARAD_1K', help='dataset name')
    parser.add_argument('--method', type=str, default='mst_plus_plus', help='reconstruction method')
    parser.add_argument('--pretrained_root', type=str, default='./RGB2HS/model_zoo', help='root path to pretrained models')
    parser.add_argument('--result_root', type=str, default='./RGB2HS/validate_exp', help='root path to validation results')
    parser.add_argument('--metamer', type=str2bool, default=False, help='indicate if data is metamer')
    parser.add_argument('--noise_type', type=str, default='poisson', choices=['poisson', 'gaussian'], help='noise type')
    parser.add_argument('--npe', type=int, default=1, help='a parameter to control the noise level')
    parser.add_argument('--rgb_format', type=str, default='jpg', choices=['png', 'jpg', 'all'], help='format for the RGB image')
    parser.add_argument('--gpu_id', type=str, default='0', help='GPU id')

    args = parser.parse_args()

    # configuration and dataset tools
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
    dtools = SourceFileLoader('dtools', './dataset_tools/loader.py').load_module()

    # data folders
    args.original_data_root = f'{args.dataset_path}/{args.dataset_name}'
    args.metamer_data_root = f'{args.original_data_root}/{args.noise_type}{args.npe}_{args.psf_name}'
    
    # pretrained model
    args.pretrained_model_path = f'{args.pretrained_root}'

    # output folder
    if args.metamer:
        args.outf = f'{args.result_root}/{args.dataset_name}_{args.noise_type}{args.npe}_{args.psf_name}_{args.rgb_format}_{args.method}_metamer/'
    else:
        args.outf = f'{args.result_root}/{args.dataset_name}_{args.noise_type}{args.npe}_{args.psf_name}_{args.rgb_format}_{args.method}_std/'
    if not os.path.exists(args.outf):
        os.makedirs(args.outf)

    # data list
    with open(f'{args.original_data_root}/split_txt/valid_list.txt', 'r') as fin:
        val_list = [line.replace('\n', '') for line in fin]
    val_list.sort()

    if args.metamer:
        print(f"Validate {args.method} on original {args.dataset_name} dataset for {args.noise_type} {args.npe} with PSF {args.psf_name}")
        # load dataset
        val_data = dtools.ValidDataset(
            original_data_root=args.original_data_root, 
            metamer_data_root=args.metamer_data_root,
            data_join="metamer",  
            bgr2rgb=True, rgb_format=args.rgb_format, method=args.method
        )
    else:
        print(f"Validate {args.method} on metamer {args.dataset_name} dataset for {args.noise_type} {args.npe} with PSF {args.psf_name}")
        # load dataset
        val_data = dtools.ValidDataset(
            original_data_root=args.original_data_root, 
            metamer_data_root=args.metamer_data_root,
            data_join="standard",  
            bgr2rgb=True, rgb_format=args.rgb_format, method=args.method
        )

    # dataloader
    val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    
    # pretrained model
    model = model_generator(args.method, args.pretrained_model_path).cuda()
    # validate
    mrae, rmse, psnr, sam = validate(val_loader, model, args, val_list, config)
    # print result
    print(f'method:{args.method}, mrae:{mrae:.4f}, rmse:{rmse:.4f}, psnr:{psnr:.2f}, sam:{sam:.4f}')
    result = {
        'metric': ['MRAE', 'RMSE', 'PSNR', 'SAM'],
        'value':[mrae.cpu().item(), rmse.cpu().item(), psnr.cpu().item(), sam.cpu().item()],
    }

    # save result
    df = pd.DataFrame(result)
    if args.metamer:
        filename = f"{args.result_root}/{args.dataset_name}_{args.noise_type}{args.npe}_{args.psf_name}_{args.rgb_format}_{args.method}_metamer.txt"
    else:
        filename = f"{args.result_root}/{args.dataset_name}_{args.noise_type}{args.npe}_{args.psf_name}_{args.rgb_format}_{args.method}_std.txt"
    df.to_csv(filename)

    print("Done")