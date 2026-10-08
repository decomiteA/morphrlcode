#!/bin/bash 
source ../local_venv/bin/activate 
python train.py baseline_lowrange_w5_seed_1 --timesteps 1_500_000_000
python train.py baseline_lowrange_w5_seed_2 --timesteps 1_500_000_000
python train.py baseline_lowrange_w5_seed_3 --timesteps 1_500_000_000
python train.py baseline_lowrange_w5_seed_4 --timesteps 1_500_000_000
python train.py baseline_lowrange_w5_seed_5 --timesteps 1_500_000_000
python train.py baseline_lowrange_w5_seed_6 --timesteps 1_500_000_000
python train.py baseline_lowrange_w5_seed_7 --timesteps 1_500_000_000
python train.py baseline_lowrange_w5_seed_8 --timesteps 1_500_000_000
