#!/bin/bash 
source ../local_venv/bin/activate 
python train.py baseline_run1
python train.py baseline_run2
python train.py baseline_run3
python train.py baseline_run4