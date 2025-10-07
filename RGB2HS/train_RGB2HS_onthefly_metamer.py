import os
import sys
sys.path.append('..')
sys.path.append('.')

import argparse
from importlib.machinery import SourceFileLoader

import numpy as np
import random
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
cudnn.benchmark = True

import metamer.utils as utils
import metamer.metrics as metrics
from metamer.utils import str2bool
from dataset_tools import loader as dtools

from architecture import *

import logging

# wavelengths
lambda1 = 0.4
lambda2 = 0.7
dlambda = 0.01
wavelengths = np.arange(lambda1, lambda2+dlambda, dlambda)

#---------- auxiliary tools ---------#
def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename

def initialize_logger(file_dir):
    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename=file_dir, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(message)s', "%Y-%m-%d %H:%M:%S")
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)
    return logger

def save_checkpoint(model_path, epoch, iteration, model, optimizer):
    state = {
        'epoch': epoch,
        'iter': iteration,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    torch.save(state, os.path.join(model_path, f'net_{epoch:d}epoch.pth'))

#---------- training and validation ---------#
def validate_step(val_loader, model, criterion, config, args):
    """
    Validate model on both standard and metamer validation data.

    Args:
        val_loader: Dataloader for the validation data.
        model: Model.
        criterion: Metric for the validation loss.
        config: Configuration of the dataset.
        args: Options.

    Returns:
        loss: Average validation loss.
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
        criterion.cuda()

    losses_mrae = metrics.AverageMeter()
    losses_rmse = metrics.AverageMeter()
    losses_psnr = metrics.AverageMeter()
    losses_sam = metrics.AverageMeter()
    losses_val = metrics.AverageMeter()

    for i, (input, target) in enumerate(val_loader):
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
        with torch.no_grad():
            if args.dataset_name == 'ICVL':
                # For ICVL, we crop the central 512 x 512 region for validation, to be consistent with others
                print("ICVL: center 512 x 512 region")
                M = config.HEIGHT // 2
                N = config.WIDTH // 2
                output = model(input[:, :, M-256:M+256, N-256:N+256])
                loss_mrae = criterion_mrae(output, target[:, :, M-256:M+256, N-256:N+256])
                loss_rmse = criterion_rmse(output, target[:, :, M-256:M+256, N-256:N+256])
                loss_psnr = criterion_psnr(output, target[:, :, M-256:M+256, N-256:N+256])
                loss_sam = criterion_sam(output, target[:, :, M-256:M+256, N-256:N+256])
                loss_val = criterion(output, target[:, :, M-256:M+256, N-256:N+256])
            else:
                # compute output
                output = model(input)
                loss_mrae = criterion_mrae(output, target)
                loss_rmse = criterion_rmse(output, target)
                loss_psnr = criterion_psnr(output, target)
                loss_sam = criterion_sam(output, target)
                loss_val = criterion(output, target)
        # record loss
        losses_mrae.update(loss_mrae.data)
        losses_rmse.update(loss_rmse.data)
        losses_psnr.update(loss_psnr.data)
        losses_sam.update(loss_sam.data)
        losses_val.update(loss_val.data)
    
    return losses_val.avg, losses_mrae.avg, losses_rmse.avg, losses_psnr.avg, losses_sam.avg

def train(train_loader, val_loader, model, args, config):
    """
    Train models on-the-fly metamers.

    Args:
        train_loader: Dataloader for training data.
        val_loader: Dataloader for validation data.
        model: Model to be trained.
        args: Options.
        config: Configuration of the dataset.
    """

    # logging
    date_time = str(datetime.datetime.now())
    date_time = time2file_name(date_time)
    log_dir = os.path.join(args.outf, f'train_{date_time}.log')
    logger = initialize_logger(log_dir)
    # header
    logger.info(f"OnTheFly metamer, {args.noise_type} noise {args.npe} for PSF {args.psf_name} for {args.method} on {args.dataset_name}")

    # wandb logging
    if args.wandb:
        import wandb
        prefix = f"OnTheFly_metamer"
        if args.pretrained_model_path is not None:
            prefix = prefix + "_pretrained"
        name = prefix + f"_{args.loss_func}_{args.noise_type}{args.npe}_{args.psf_name}_{args.method}_{args.dataset_name}"
        if not args.on_the_fly:
            print("No on-the-fly metameric augmentation")
            name = name + "_noaug"
        print(f"Starting wandb run: {name}")

        try:
            # replace 'your_project' with your actual project name, and
            # replace 'your_entity' with your actual wandb entity name
            run = wandb.init(project="your_project", entity='your_entity', name=name) 
            
            wandb.config.update({
                "dataset_name": args.dataset_name,
                "pretrained_model_path": args.pretrained_model_path,
                "batch_size": args.batch_size,
                "noise_type": args.noise_type,
                "npe": args.npe,
                "psf_name": args.psf_name,
                "rgb_format": args.rgb_format,
                "method": args.method,
                "loss_func": args.loss_func,
                "on_the_fly": args.on_the_fly,
            })

            wandb.watch(model)

        except Exception as e:
            print(f"wandb not available or login failed, continuing without it: {e}")
            args.wandb = False
    
    # load SRF data
    SRF = utils.load_srf(args.srf_path, dtype=np.float32)
    SRF = torch.from_numpy(SRF)
    try:
        PSFs = np.load(f"{args.psf_path}/{args.psf_name}.npz")['PSFs'].astype(np.float32)
        PSFs = torch.from_numpy(PSFs.transpose(2,0,1)) # [H, W, C] -> [C, H, W]
    except:
        if (str(args.psf_name).lower() == 'none') or (args.psf_name is None):
            PSFs = None
        else:
            raise ValueError(f"No such file {args.psf_path} for PSF.")

    # iterations
    per_epoch_iteration = args.per_epoch_iteration
    total_iteration = per_epoch_iteration * args.num_epochs

    optimizer = optim.Adam(model.parameters(), lr=args.init_lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iteration, eta_min=1e-6)

    if args.pretrained_model_path is not None:
        print(f"Load checkpoint '{args.pretrained_model_path}'")
        checkpoint = torch.load(args.pretrained_model_path)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch'] if 'epoch' in checkpoint.keys() else 0
        iteration = checkpoint['iter'] if 'epoch' in checkpoint.keys() else 0
        if 'optimizer' in checkpoint.keys():
            optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        start_epoch = 0
        iteration = 0
    
    # loss functions for evaluation
    criterion_mrae = metrics.Loss_MRAE()
    criterion_rmse = metrics.Loss_RMSE()
    criterion_psnr = metrics.Loss_PSNR()
    criterion_sam = metrics.Loss_SAM()

    # training loss function
    if args.loss_func == 'l1':
        criterion = nn.L1Loss()
    elif args.loss_func == 'l2':
        criterion = nn.MSELoss()
    elif args.loss_func == 'mrae':
        criterion = criterion_mrae
    elif args.loss_func == 'rmse':
        criterion = criterion_rmse
    elif args.loss_func == 'psnr':
        criterion = criterion_psnr
    elif args.loss_func == 'sam':
        criterion = criterion_sam
    else:
        raise ValueError(f"Unknown loss function {args.loss_func}.")
    
    # move all to cuda
    if torch.cuda.is_available():
        model = model.cuda()
        SRF = SRF.cuda()
        if PSFs is not None:
            PSFs = PSFs.cuda()
        criterion_mrae.cuda()
        criterion_rmse.cuda()
        criterion_psnr.cuda()
        criterion_sam.cuda()
        criterion.cuda()
    
    # validation before training
    print("Validation before training")
    val_loss, mrae_loss, rmse_loss, psnr_loss, sam_loss = validate_step(val_loader, model, criterion, config, args)
    print(f'Validation loss {val_loss}, MRAE:{mrae_loss}, RMSE: {rmse_loss}, PNSR:{psnr_loss}, SAM:{sam_loss}')
    logger.info(f"Iter[{iteration:06d}], Epoch[{iteration//per_epoch_iteration:06d}], learning rate: N/A, Validation Loss: {val_loss:.9f}, Test MRAE: {mrae_loss:.9f}, "
                f"Test RMSE: {rmse_loss:.9f}, Test PSNR: {psnr_loss:.9f} Test SAM: {sam_loss:.9f}")
    if args.wandb:
        wandb.log(
            {'validation_loss/loss': val_loss, 'val_mrae': mrae_loss, 'val_rmse': rmse_loss, 'val_psnr': psnr_loss, 'val_sam': sam_loss}, step=iteration
        )
    
    # training loop
    recorded_loss = 1000
    while iteration < total_iteration:
        losses = metrics.AverageMeter()
        for i, (bgr, hsi) in enumerate(train_loader):
            model.train()

            if torch.cuda.is_available():
                hsi = hsi.cuda()
                bgr = bgr.cuda()

            lr = optimizer.param_groups[0]['lr']
            optimizer.zero_grad()

            # reconstruction
            output = model(bgr)
            loss = criterion(output, hsi)
            loss.backward()
            
            # update
            optimizer.step()
            scheduler.step()
            losses.update(loss.data)

            iteration = iteration + 1

            # logging
            if iteration % args.log_freq == 0:
                print(f'[iter:{iteration}/{total_iteration}],lr={lr:.9f}, train_losses.avg={losses.avg:.9f}')
                logger.info(f'[iter:{iteration}/{total_iteration}],lr={lr:.9f}, train_losses.avg={losses.avg:.9f}')
                if args.wandb:
                    wandb.log({'train_loss/loss': losses.avg}, step=iteration)
            
            if iteration % args.valid_freq == 0:
                # validation
                val_loss, mrae_loss, rmse_loss, psnr_loss, sam_loss = validate_step(val_loader, model, criterion, config, args)
                print(f'Validation loss {val_loss}, MRAE:{mrae_loss}, RMSE: {rmse_loss}, PNSR:{psnr_loss}, SAM:{sam_loss}')
                # print loss
                print(f" Iter[{iteration:06d}], Epoch[{iteration//per_epoch_iteration:06d}], learning rate: {lr:.9f}, Train Loss: {losses.avg:.9f}, Test MRAE: {mrae_loss:.9f}, "
                      f"Test RMSE: {rmse_loss:.9f}, Test PSNR: {psnr_loss:.9f} Test SAM: {sam_loss:.9f}")
                logger.info(f" Iter[{iteration:06d}], Epoch[{iteration//per_epoch_iteration:06d}], learning rate : {lr:.9f}, Train Loss: {losses.avg:.9f}, Test MRAE: {mrae_loss:.9f}, "
                      f"Test RMSE: {rmse_loss:.9f}, Test PSNR: {psnr_loss:.9f} Test SAM: {sam_loss:.9f}")
                if args.wandb:
                    wandb.log(
                        {'validation_loss/loss': val_loss, 'val_mrae': mrae_loss, 'val_rmse': rmse_loss, 'val_psnr': psnr_loss, 'val_sam': sam_loss}, step=iteration
                    )

                if torch.abs(val_loss - recorded_loss) < 0.01 or val_loss < recorded_loss or iteration % args.save_freq == 0:
                    print("----------")
                    print(f'Saving to {args.outf} epoch {iteration // per_epoch_iteration}')
                    save_checkpoint(args.outf, (iteration // per_epoch_iteration), iteration, model, optimizer)
                    if val_loss < recorded_loss:
                        recorded_loss = val_loss
                    print("----------")
            

if __name__ == "__main__":
    """
    Train models with on-the-fly metamers.
    """

    parser = argparse.ArgumentParser(
        description='RGB2HS model training with on-the-fly metamers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--srf_path', type=str, default='./resources/A2a5320-23ucBAS_gain.npz',help='path to spectral response function')
    parser.add_argument('--psf_path', type=str, default='./PSFs', help='path to point spread function')
    parser.add_argument('--psf_name', type=str, default=None, help='point spread function name')
    parser.add_argument('--dataset_path', type=str, default='./datasets', help='path to datasets')
    parser.add_argument('--dataset_name', type=str, default='ARAD_1K', help='dataset name')
    parser.add_argument('--method', type=str, default='mst_plus_plus', help='reconstruction method')
    parser.add_argument('--pretrained_root', type=str, default=None, help='root path to pretrained models')
    parser.add_argument('--resume_file', type=str, default=None, help='continue from previous pretrained models')
    parser.add_argument('--result_root', type=str, default='./RGB2HS/train_exp_onthefly_metamer', help='root path to training results')
    parser.add_argument('--noise_type', type=str, default='poisson', choices=['poisson', 'gaussian'], help='noise type')
    parser.add_argument('--npe', type=int, default=0, help='a parameter to control the noise level')
    parser.add_argument('--rgb_format', type=str, default='png', choices=['png', 'jpg'], help='format of the RGB image')
    parser.add_argument('--num_epochs', type=int, default=300, help='number of epochs')
    parser.add_argument('--init_lr', type=float, default=4e-4, help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=20, help='batch size for training')
    parser.add_argument('--crop_size', type=int, default=128, help='crop size (patch size) for training')
    parser.add_argument('--stride', type=int, default=8, help='stride size to pick up data')
    parser.add_argument('--per_epoch_iteration', type=int, default=1000, help='per epoch iterations')
    parser.add_argument('--loss_func', type=str, default='l1', choices=['l1', 'mrae', 'rmse', 'psnr', 'sam'], help='loss function for training')
    parser.add_argument('--log_freq', type=int, default=20, help='frequency of logging')
    parser.add_argument('--valid_freq', type=int, default=100, help='frequency of validation')
    parser.add_argument('--save_freq', type=int, default=1000, help='frequency of saving checkpoints')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--gpu_id', type=str, default='0', help='GPU id')
    parser.add_argument('--data_join', type=str, default="standard", help='both, metamer, standard')
    parser.add_argument('--on_the_fly', type=str2bool, default=True, help='metamer pair generated during training')
    parser.add_argument('--wandb', type=str2bool, default=False, help='wandb logging')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.outf = f"{args.result_root}/OnTheFly_metamer_{args.loss_func}_{args.noise_type}{args.npe}_{args.psf_name}_{args.method}_{args.dataset_name}"
    if not os.path.exists(args.outf):
        os.makedirs(args.outf)
    
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
    
    # data folders
    args.original_data_root = f'{args.dataset_path}/{args.dataset_name}'
    args.metamer_data_root = f'{args.original_data_root}/{args.noise_type}{args.npe}_{args.psf_name}'

    # load original training data
    print("Load training data")
    train_data = dtools.TrainDataset(
        original_data_root=args.original_data_root, 
        metamer_data_root=args.metamer_data_root,
        crop_size=args.crop_size, stride=args.stride,
        data_join="standard", # must be "standard" for on-the-fly metamers
        bgr2rgb=True, 
        aug=True,
        on_the_fly=args.on_the_fly, # enables metamer data generation on-the-fly if True
        args=args,
        config=config
    )
    
    # load validation data (standard + metamer)
    print("Load validation data")
    val_data = dtools.ValidDataset(
        original_data_root=args.original_data_root, 
        metamer_data_root=args.metamer_data_root,
        data_join="both", # must be "both" for on-the-fly metamers 
        bgr2rgb=True,
    )

    # training dataloader
    train_loader = DataLoader(
        dataset=train_data, batch_size=args.batch_size, shuffle=True, 
        num_workers=2, pin_memory=True, drop_last=True
    )

    # validation dataloader
    val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    # generate model
    if (args.resume_file is not None) or str(args.resume_file).lower() == 'none':
        args.pretrained_model_path = None
    else:
        args.pretrained_model_path = f'{args.pretrained_root}/{args.resume_file}.pth'
    
    # generate a Null model first
    model = model_generator(args.method, None)
    print('   Number of parameters: ', sum(param.numel() for param in model.parameters()))

    print("Start training with on-the-fly metamers")
    train(train_loader, val_loader, model, args, config)

    print(f"Done!")