#!/bin/bash 
source ../local_venv/bin/activate 
python train.py baseline_lowrange_d5_3b_seed_1 --timesteps 3_000_000_000
python train.py baseline_lowrange_d5_3b_seed_2 --timesteps 3_000_000_000
python train.py baseline_lowrange_d5_3b_seed_3 --timesteps 3_000_000_000
python train.py baseline_lowrange_d5_3b_seed_4 --timesteps 3_000_000_000
python train.py baseline_lowrange_d5_3b_seed_5 --timesteps 3_000_000_000
python train.py baseline_lowrange_d5_3b_seed_6 --timesteps 3_000_000_000
python train.py baseline_lowrange_d5_3b_seed_7 --timesteps 3_000_000_000
python train.py baseline_lowrange_d5_3b_seed_8 --timesteps 3_000_000_000
