import torch
import torch.nn as nn
import argparse
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
from hsi_dataset import TrainDataset, ValidDataset
from architecture import *
from utils import AverageMeter, initialize_logger, save_checkpoint, record_loss, \
    time2file_name, Loss_MRAE, Loss_RMSE, Loss_PSNR, Loss_SAM, load_srf
import datetime
import numpy as np
from utils import learning_rates_dict
import time

def str2bool(v):
    return v.lower() in ('yes', 'y', 'true', 't', '1')

parser = argparse.ArgumentParser(description="Spectral Recovery Toolbox")
parser.add_argument('--method', type=str, default='mst_plus_plus')
parser.add_argument('--pretrained_model_path', type=str, default=None)
parser.add_argument("--percent", type=lambda x: int(x) if x.lower() != 'none' else None, default=None, help="percentage of training data")
parser.add_argument("--batch_size", type=int, default=20, help="batch size")
parser.add_argument("--end_epoch", type=int, default=300, help="number of epochs")
parser.add_argument("--init_lr", type=float, default=1e-4, help="initial learning rate")
parser.add_argument("--outf", type=str, default='./exp/mst_plus_plus/', help='path log files')
parser.add_argument("--data_root", type=str, default='../dataset/')
parser.add_argument("--patch_size", type=int, default=128, help="patch size")
parser.add_argument("--stride", type=int, default=8, help="stride")
parser.add_argument("--gpu_id", type=str, default='0', help='path log files')
parser.add_argument('--wandb', type=str2bool, default=False, help='wandb logging')
opt = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

# load dataset
print("\nloading dataset ...")
if opt.percent is not None:
    if (opt.percent < 10) or (opt.percent > 100):
        raise ValueError(f"Percentage of data should be [10, 100], but got {opt.percent}")
    print(f"using {opt.percent}% of the dataset for training")
train_data = TrainDataset(data_root=opt.data_root, crop_size=opt.patch_size, bgr2rgb=True, arg=True, stride=opt.stride, percent=opt.percent, method=opt.method)
print(f"Iteration per epoch: {len(train_data)}")
val_data = ValidDataset(data_root=opt.data_root, bgr2rgb=True, method=opt.method)
print("Validation set samples: ", len(val_data))

# iterations
per_epoch_iteration = 1000
total_iteration = per_epoch_iteration*opt.end_epoch

# loss function
criterion_mrae = Loss_MRAE()
criterion_rmse = Loss_RMSE()
criterion_psnr = Loss_PSNR()
criterion_sam = Loss_SAM()

# model
if str(opt.pretrained_model_path).lower() == 'none':
    opt.pretrained_model_path = None
pretrained_model_path = opt.pretrained_model_path
method = opt.method
init_lr = learning_rates_dict[method]
model = model_generator(method, pretrained_model_path).cuda()
print('Parameters number is ', sum(param.numel() for param in model.parameters()))

if opt.wandb:
    import wandb
    name = "vanilla_" + opt.method
    if opt.percent is not None:
        name = name + f"_{opt.percent}percent"
    if opt.pretrained_model_path is not None:
        name = name + "_pretrained"
    print(f"Starting wandb run: {name}")
    try:
        # replace 'your_project' with your actual project name, and
        # replace 'your_entity' with your actual wandb entity name
        run = wandb.init(project="your_project", entity='your_entity', name=name) 

        wandb.config.update({
            "method": opt.method,
            "pretrained_model_path": opt.pretrained_model_path,
            "batch_size": opt.batch_size,
            "end_epoch": opt.end_epoch,
            "init_lr": init_lr,
            "patch_size": opt.patch_size,
            "strid": opt.stride,
        })

        wandb.watch(model)

    except Exception as e:
        print(f"wandb not available or login failed, continuing without it: {e}")
        opt.wandb = False

# output path
# date_time = str(datetime.datetime.now())
# date_time = time2file_name(date_time)
if opt.percent is not None:
    opt.outf = opt.outf + f"_{opt.percent}percent"
if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

if torch.cuda.is_available():
    model.cuda()
    criterion_mrae.cuda()
    criterion_rmse.cuda()
    criterion_psnr.cuda()
    criterion_sam.cuda()

optimizer = optim.Adam(model.parameters(), lr=init_lr, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iteration, eta_min=1e-6)

# logging
if opt.percent is not None:
    log_dir = os.path.join(opt.outf, f'train_{opt.percent}percent.log')
else:
    log_dir = os.path.join(opt.outf, 'train_full.log')
logger = initialize_logger(log_dir)

# Resume
resume_file = opt.pretrained_model_path
if resume_file is not None:
    if os.path.isfile(resume_file):
        print("=> loading checkpoint '{}'".format(resume_file))
        checkpoint = torch.load(resume_file)
        start_epoch = checkpoint['epoch'] if 'epoch' in checkpoint.keys() else 0
        iteration = checkpoint['iter'] if 'iter' in checkpoint.keys() else 0
        model.load_state_dict(checkpoint['state_dict'])
        if 'optimizer' in checkpoint.keys():
            optimizer.load_state_dict(checkpoint['optimizer'])

def main():
    cudnn.benchmark = True
    iteration = 0

    train_loader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    # validation before training
    print("Validation before training")
    mrae_loss, rmse_loss, psnr_loss, sam_loss = validate(val_loader, model)
    print(f'MRAE:{mrae_loss}, RMSE: {rmse_loss}, PNSR:{psnr_loss}, SAM: {sam_loss}')
    if opt.wandb:
        wandb.log(
            {'val_mrae': mrae_loss, 'val_rmse': rmse_loss, 'val_psnr': psnr_loss, 'val_sam': sam_loss},step=0
        )

    record_mrae_loss = 1000
    while iteration<total_iteration:
        model.train()
        losses = AverageMeter()
        for i, (images, labels) in enumerate(train_loader):
            labels = labels.cuda()
            labels = Variable(labels)
            lr = optimizer.param_groups[0]['lr']
            optimizer.zero_grad()

            if opt.method == 'hprn':
                semantic_labels = images[1].cuda()
                semantic_labels = Variable(semantic_labels)
                images = images[0].cuda()
                images = Variable(images)
                output = model(images, semantic_labels.to(images.device))
                loss = criterion_mrae(output, labels)
            elif opt.method == 'ssrnet':
                images = images[0].cuda()
                images = Variable(images)
                RGB_FILTER_CSV = './resources/A2a5320-23ucBAS_gain.npz'
                SRF = load_srf(RGB_FILTER_CSV, dtype=np.float32)
                SRF = torch.from_numpy(SRF).permute(1, 0)
                SRF_normalized = (SRF/SRF.sum(-1, keepdims=True))
                output = model(SRF_normalized.to(images.device), images)
                loss = criterion_mrae(output, labels)
            else:
                images = images.cuda()
                output = model(images)
                loss = criterion_mrae(output, labels)


            loss.backward()
            optimizer.step()
            scheduler.step()
            
            losses.update(loss.data)
            iteration = iteration+1
            
            if iteration % 20 == 0:
                print('[iter:%d/%d],lr=%.9f,train_losses.avg=%.9f'
                      % (iteration, total_iteration, lr, losses.avg))
                if opt.wandb:
                    wandb.log({'train_loss/loss': losses.avg},step=iteration)
            if iteration % 1000 == 0:
                mrae_loss, rmse_loss, psnr_loss, sam_loss = validate(val_loader, model)
                print(f'MRAE:{mrae_loss}, RMSE: {rmse_loss}, PNSR:{psnr_loss}, SAM: {sam_loss}')
                # Save model
                if torch.abs(mrae_loss - record_mrae_loss) < 0.01 or mrae_loss < record_mrae_loss or iteration % 5000 == 0:
                    print(f'Saving to {opt.outf}')
                    save_checkpoint(opt.outf, (iteration // 1000), iteration, model, optimizer)
                    if mrae_loss < record_mrae_loss:
                        record_mrae_loss = mrae_loss
                # print loss
                print(" Iter[%06d], Epoch[%06d], learning rate : %.9f, Train MRAE: %.9f, Test MRAE: %.9f, "
                      "Test RMSE: %.9f, Test PSNR: %.9f, Test SAM: %.9f" % (iteration, iteration//1000, lr, losses.avg, mrae_loss, rmse_loss, psnr_loss, sam_loss))
                logger.info(" Iter[%06d], Epoch[%06d], learning rate : %.9f, Train Loss: %.9f, Test MRAE: %.9f, "
                      "Test RMSE: %.9f, Test PSNR: %.9f, Test SAM: %.9f" % (iteration, iteration//1000, lr, losses.avg, mrae_loss, rmse_loss, psnr_loss, sam_loss))
                if opt.wandb:
                    wandb.log(
                        {'val_mrae': mrae_loss, 'val_rmse': rmse_loss, 'val_psnr': psnr_loss, 'val_sam': sam_loss},step=iteration
                    )
    return 0

# Validate
def validate(val_loader, model):
    model.eval()
    losses_mrae = AverageMeter()
    losses_rmse = AverageMeter()
    losses_psnr = AverageMeter()
    losses_sam = AverageMeter()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        with torch.no_grad():
            # compute output
            if opt.method == 'hprn':
                semantic_labels = input[1].cuda()
                input = input[0].cuda()
                output = model(input, semantic_labels)
            elif opt.method == 'ssrnet':
                input = input.cuda()
                RGB_FILTER_CSV = './resources/A2a5320-23ucBAS_gain.npz'
                SRF = load_srf(RGB_FILTER_CSV, dtype=np.float32)
                SRF = torch.from_numpy(SRF).permute(1, 0)
                SRF_normalized = (SRF/SRF.sum(-1, keepdims=True))
                output = model(SRF_normalized.to(input.device), input)
            else:
                input = input.cuda()
                output = model(input)

            # Note: the original MST++ paper evaluates the central part only
            # loss_mrae = criterion_mrae(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
            # loss_rmse = criterion_rmse(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
            # loss_psnr = criterion_psnr(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
            # Note: we evaluate the full datacube
            loss_mrae = criterion_mrae(output, target)
            loss_rmse = criterion_rmse(output, target)
            loss_psnr = criterion_psnr(output, target)
            loss_sam = criterion_sam(output, target)
        # record loss
        losses_mrae.update(loss_mrae.data)
        losses_rmse.update(loss_rmse.data)
        losses_psnr.update(loss_psnr.data)
        losses_sam.update(loss_sam.data)
    return losses_mrae.avg, losses_rmse.avg, losses_psnr.avg, losses_sam.avg

if __name__ == '__main__':
    main()
    print(torch.__version__)