#!/bin/sh
cd /root/gsc/offlinerl_experiments/OfflineRL-Kit;

# hopper-random-v2
CUDA_VISIBLE_DEVICES=0 python run_example/run_cql.py --task "hopper-random-v2" --seed 33 --epoch 200 --step-per-epoch 250  & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_iql.py --task "hopper-random-v2" --seed 33 --epoch 200 --step-per-epoch 250  & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_td3bc.py --task "hopper-random-v2" --seed 33 --epoch 200 --step-per-epoch 250  & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_mcq.py --task "hopper-random-v2" --seed 33 --epoch 200 --step-per-epoch 250  & 

CUDA_VISIBLE_DEVICES=0 python run_example/run_cql.py --task "hopper-random-v2" --seed 66 --epoch 200 --step-per-epoch 250  & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_iql.py --task "hopper-random-v2" --seed 66 --epoch 200 --step-per-epoch 250  & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_td3bc.py --task "hopper-random-v2" --seed 66 --epoch 200 --step-per-epoch 250  & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_mcq.py --task "hopper-random-v2" --seed 66 --epoch 200 --step-per-epoch 250  & 

CUDA_VISIBLE_DEVICES=0 python run_example/run_cql.py --task "hopper-random-v2" --seed 99 --epoch 200 --step-per-epoch 250  & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_iql.py --task "hopper-random-v2" --seed 99 --epoch 200 --step-per-epoch 250  & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_td3bc.py --task "hopper-random-v2" --seed 99 --epoch 200 --step-per-epoch 250  & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_mcq.py --task "hopper-random-v2" --seed 99 --epoch 200 --step-per-epoch 250  & 


CUDA_VISIBLE_DEVICES=1 python run_example/run_cql.py --task "hopper-random-v2" --seed 33 --epoch 200 --step-per-epoch 250 --interaction_tag  & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_iql.py --task "hopper-random-v2" --seed 33 --epoch 200 --step-per-epoch 250 --interaction_tag  & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_td3bc.py --task "hopper-random-v2" --seed 33 --epoch 200 --step-per-epoch 250 --interaction_tag  & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_mcq.py --task "hopper-random-v2" --seed 33 --epoch 200 --step-per-epoch 250 --interaction_tag  & 

CUDA_VISIBLE_DEVICES=1 python run_example/run_cql.py --task "hopper-random-v2" --seed 66 --epoch 200 --step-per-epoch 250 --interaction_tag  & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_iql.py --task "hopper-random-v2" --seed 66 --epoch 200 --step-per-epoch 250 --interaction_tag  & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_td3bc.py --task "hopper-random-v2" --seed 66 --epoch 200 --step-per-epoch 250 --interaction_tag  & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_mcq.py --task "hopper-random-v2" --seed 66 --epoch 200 --step-per-epoch 250 --interaction_tag  & 

CUDA_VISIBLE_DEVICES=1 python run_example/run_cql.py --task "hopper-random-v2" --seed 99 --epoch 200 --step-per-epoch 250 --interaction_tag  & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_iql.py --task "hopper-random-v2" --seed 99 --epoch 200 --step-per-epoch 250 --interaction_tag  & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_td3bc.py --task "hopper-random-v2" --seed 99 --epoch 200 --step-per-epoch 250 --interaction_tag  & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_mcq.py --task "hopper-random-v2" --seed 99 --epoch 200 --step-per-epoch 250 --interaction_tag  & 