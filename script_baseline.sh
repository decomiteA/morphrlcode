#!/bin/bash 
source ../local_venv/bin/activate 
python train.py long_legs_2 --timesteps 1_500_000_000
python train.py long_legs_3 --timesteps 1_500_000_000
python train.py long_legs_4 --timesteps 1_500_000_000
python train.py long_legs_5 --timesteps 1_500_000_000
python train.py long_legs_6 --timesteps 1_500_000_000
python train.py long_legs_7 --timesteps 1_500_000_000
python train.py long_legs_8 --timesteps 1_500_000_000
