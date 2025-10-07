#!/bin/bash
# =========================================================
#  Train models with on-the-fly metamers.
#  Tested 17 methods: MST++, MST, MPRNet, Restormer,  
#  MIRNet, HINet, HDNet, AWAN, EDSR, HRNet, HSCNN+, HySAT, 
#  HPRN, SSTHyper, MSFN, GMSR, SSRNet.
#  Tested 4 datasets: ARAD_1K, CAVE, ICVL, KAUST.
# =========================================================

PSF_NAME=None
DATASET_PATH=./datasets
DATASET_NAME=ARAD_1K
METHOD=mst_plus_plus 
RESULT_ROOT=./RGB2HS/train_exp_onthefly_metamer
NOISE_TYPE=poisson
NPE=0
RGB_FORMAT=png
INIT_LR=4e-4
CROP_SIZE=128
STRIDE=8
NUM_EPOCHS=300
BATCH_SIZE=20
LOSS_FUNC=l1
WANDB=False

python ./RGB2HS/train_RGB2HS_onthefly_metamer.py \
--psf_name $PSF_NAME \
--dataset_path $DATASET_PATH \
--dataset_name $DATASET_NAME \
--method $METHOD \
--result_root $RESULT_ROOT \
--noise_type $NOISE_TYPE \
--npe $NPE \
--rgb_format $RGB_FORMAT \
--num_epochs $NUM_EPOCHS \
--init_lr $INIT_LR \
--batch_size $BATCH_SIZE \
--crop_size $CROP_SIZE \
--stride $STRIDE \
--loss_func $LOSS_FUNC \
--wandb $WANDB \