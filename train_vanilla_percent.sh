#!/usr/bin/env bash
# =========================================================
#  Train vanilla models with various dataset percentages.
#  Tested 17 methods: MST++, MST, MPRNet, Restormer,  
#  MIRNet, HINet, HDNet, AWAN, EDSR, HRNet, HSCNN+, HySAT, 
#  HPRN, SSTHyper, MSFN, GMSR, SSRNet.
#  Tested 4 datasets: ARAD_1K, CAVE, ICVL, KAUST.
#  Tested 3 percentages: 100% (None), 50%, and 20%.
# =========================================================

# ---------- Configuration ----------
METHOD="mst_plus_plus"
PRETRAINED_MODEL_PATH=None      # Path to checkpoint, or "None"
PERCENT=50                      # Use None, 50, 20 to control dataset percentage
BATCH_SIZE=20
END_EPOCH=300
INIT_LR=4e-4                    # Initial learning rate, see README.md for details
OUTF=./MST_plus_plus_code/train_exp/
DATA_ROOT=./datasets/ARAD_1K/  
PATCH_SIZE=128
STRIDE=8
GPU_ID=0
WANDB=False                     # Set True to enable WandB logging

python ./MST_plus_plus_code/train_code/train.py \
--method $METHOD \
--pretrained_model_path $PRETRAINED_MODEL_PATH \
--percent $PERCENT \
--batch_size $BATCH_SIZE \
--end_epoch $END_EPOCH \
--init_lr $INIT_LR \
--outf $OUTF \
--data_root $DATA_ROOT \
--patch_size $PATCH_SIZE \
--stride $STRIDE \
--gpu_id $GPU_ID \
--wandb $WANDB