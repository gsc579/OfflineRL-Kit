#!/bin/sh
cd /root/gsc/offlinerl_experiments/OfflineRL-Kit;

CUDA_VISIBLE_DEVICES=0 python run_example/run_cql.py --task "pen-human-v1" --seed 33 --epoch 200 --step-per-epoch 250 --interaction_tag & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_cql.py --task "pen-human-v1" --seed 33 --epoch 200 --step-per-epoch 250 &