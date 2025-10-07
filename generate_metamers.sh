#!/bin/bash
# =========================================================
#  Generate metamer data from existing datasets.
#  Tested 4 datasets: ARAD_1K, CAVE, ICVL, KAUST.
# =========================================================

python ./metamer/generate_metamers.py \
--psf_name None \
--dataset_path /home/fuq/projects/Github/MetamerAdversary/datasets \
--dataset_name ARAD_1K \
--npe 0 \
--alpha 0.0 \
--rgb_format png