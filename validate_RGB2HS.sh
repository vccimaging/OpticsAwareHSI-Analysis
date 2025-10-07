#!/bin/bash
# =========================================================
#  Validate pretrained models with unseen data.
# =========================================================

PSF_NAME=None
DATASET_PATH=./datasets
DATASET_NAME=ARAD_1K
METHOD=mst_plus_plus
PRETRAINED_ROOT=./RGB2HS/model_zoo/mst_plus_plus.pth
RESULT_ROOT=./RGB2HS/validate_exp 
METAMER=False
NOISE_TYPE=poisson
NPE=0 
RGB_FORMAT=png
GPU_ID=0

python ./RGB2HS/validate_RGB2HS.py \
--psf_name $PSF_NAME \
--dataset_path $DATASET_PATH \
--dataset_name $DATASET_NAME \
--method $METHOD \
--pretrained_root $PRETRAINED_ROOT \
--result_root $RESULT_ROOT \
--metamer $METAMER \
--noise_type $NOISE_TYPE \
--npe $NPE \
--rgb_format $RGB_FORMAT \
--gpu_id $GPU_ID