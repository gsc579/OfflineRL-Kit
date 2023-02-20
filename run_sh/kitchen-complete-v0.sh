#!/bin/sh
cd /root/gsc/offlinerl_experiments/OfflineRL-Kit;

# kitchen-complete-v0
CUDA_VISIBLE_DEVICES=0 python run_example/run_cql.py --task "kitchen-complete-v0" --seed 33 --epoch 200 --step-per-epoch 250  & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_iql.py --task "kitchen-complete-v0" --seed 33 --epoch 200 --step-per-epoch 250  & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_td3bc.py --task "kitchen-complete-v0" --seed 33 --epoch 200 --step-per-epoch 250  & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_mcq.py --task "kitchen-complete-v0" --seed 33 --epoch 200 --step-per-epoch 250  & 

CUDA_VISIBLE_DEVICES=0 python run_example/run_cql.py --task "kitchen-complete-v0" --seed 66 --epoch 200 --step-per-epoch 250  & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_iql.py --task "kitchen-complete-v0" --seed 66 --epoch 200 --step-per-epoch 250  & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_td3bc.py --task "kitchen-complete-v0" --seed 66 --epoch 200 --step-per-epoch 250  & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_mcq.py --task "kitchen-complete-v0" --seed 66 --epoch 200 --step-per-epoch 250  & 

CUDA_VISIBLE_DEVICES=0 python run_example/run_cql.py --task "kitchen-complete-v0" --seed 99 --epoch 200 --step-per-epoch 250  & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_iql.py --task "kitchen-complete-v0" --seed 99 --epoch 200 --step-per-epoch 250  & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_td3bc.py --task "kitchen-complete-v0" --seed 99 --epoch 200 --step-per-epoch 250  & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_mcq.py --task "kitchen-complete-v0" --seed 99 --epoch 200 --step-per-epoch 250  & 


CUDA_VISIBLE_DEVICES=1 python run_example/run_cql.py --task "kitchen-complete-v0" --seed 33 --epoch 200 --step-per-epoch 250 --interaction_tag  & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_iql.py --task "kitchen-complete-v0" --seed 33 --epoch 200 --step-per-epoch 250 --interaction_tag  & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_td3bc.py --task "kitchen-complete-v0" --seed 33 --epoch 200 --step-per-epoch 250 --interaction_tag  & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_mcq.py --task "kitchen-complete-v0" --seed 33 --epoch 200 --step-per-epoch 250 --interaction_tag  & 

CUDA_VISIBLE_DEVICES=1 python run_example/run_cql.py --task "kitchen-complete-v0" --seed 66 --epoch 200 --step-per-epoch 250 --interaction_tag  & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_iql.py --task "kitchen-complete-v0" --seed 66 --epoch 200 --step-per-epoch 250 --interaction_tag  & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_td3bc.py --task "kitchen-complete-v0" --seed 66 --epoch 200 --step-per-epoch 250 --interaction_tag  & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_mcq.py --task "kitchen-complete-v0" --seed 66 --epoch 200 --step-per-epoch 250 --interaction_tag  & 

CUDA_VISIBLE_DEVICES=1 python run_example/run_cql.py --task "kitchen-complete-v0" --seed 99 --epoch 200 --step-per-epoch 250 --interaction_tag  & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_iql.py --task "kitchen-complete-v0" --seed 99 --epoch 200 --step-per-epoch 250 --interaction_tag  & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_td3bc.py --task "kitchen-complete-v0" --seed 99 --epoch 200 --step-per-epoch 250 --interaction_tag  & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_mcq.py --task "kitchen-complete-v0" --seed 99 --epoch 200 --step-per-epoch 250 --interaction_tag  & 