#!/bin/sh
cd /root/gsc/offlinerl_experiments/OfflineRL-Kit;

# pen-human-v1
CUDA_VISIBLE_DEVICES=0 python run_example/run_cql.py --task "pen-human-v1" --seed 33 --epoch 200 --step-per-epoch 250 --interaction_tag "True" & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_iql.py --task "pen-human-v1" --seed 33 --epoch 200 --step-per-epoch 250 --interaction_tag True & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_td3bc.py --task "pen-human-v1" --seed 33 --epoch 200 --step-per-epoch 250 --interaction_tag True & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_mcq.py --task "pen-human-v1" --seed 33 --epoch 200 --step-per-epoch 250 --interaction_tag True & 

CUDA_VISIBLE_DEVICES=0 python run_example/run_cql.py --task "pen-human-v1" --seed 66 --epoch 200 --step-per-epoch 250 --interaction_tag True & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_iql.py --task "pen-human-v1" --seed 66 --epoch 200 --step-per-epoch 250 --interaction_tag True & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_td3bc.py --task "pen-human-v1" --seed 66 --epoch 200 --step-per-epoch 250 --interaction_tag True & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_mcq.py --task "pen-human-v1" --seed 66 --epoch 200 --step-per-epoch 250 --interaction_tag True & 

CUDA_VISIBLE_DEVICES=0 python run_example/run_cql.py --task "pen-human-v1" --seed 99 --epoch 200 --step-per-epoch 250 --interaction_tag True & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_iql.py --task "pen-human-v1" --seed 99 --epoch 200 --step-per-epoch 250 --interaction_tag True & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_td3bc.py --task "pen-human-v1" --seed 99 --epoch 200 --step-per-epoch 250 --interaction_tag True & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_mcq.py --task "pen-human-v1" --seed 99 --epoch 200 --step-per-epoch 250 --interaction_tag True & 


CUDA_VISIBLE_DEVICES=1 python run_example/run_cql.py --task "pen-human-v1" --seed 33 --epoch 200 --step-per-epoch 250 --interaction_tag False & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_iql.py --task "pen-human-v1" --seed 33 --epoch 200 --step-per-epoch 250 --interaction_tag False & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_td3bc.py --task "pen-human-v1" --seed 33 --epoch 200 --step-per-epoch 250 --interaction_tag False & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_mcq.py --task "pen-human-v1" --seed 33 --epoch 200 --step-per-epoch 250 --interaction_tag False & 

CUDA_VISIBLE_DEVICES=1 python run_example/run_cql.py --task "pen-human-v1" --seed 66 --epoch 200 --step-per-epoch 250 --interaction_tag False & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_iql.py --task "pen-human-v1" --seed 66 --epoch 200 --step-per-epoch 250 --interaction_tag False & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_td3bc.py --task "pen-human-v1" --seed 66 --epoch 200 --step-per-epoch 250 --interaction_tag False & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_mcq.py --task "pen-human-v1" --seed 66 --epoch 200 --step-per-epoch 250 --interaction_tag False & 

CUDA_VISIBLE_DEVICES=1 python run_example/run_cql.py --task "pen-human-v1" --seed 99 --epoch 200 --step-per-epoch 250 --interaction_tag False & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_iql.py --task "pen-human-v1" --seed 99 --epoch 200 --step-per-epoch 250 --interaction_tag False & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_td3bc.py --task "pen-human-v1" --seed 99 --epoch 200 --step-per-epoch 250 --interaction_tag False & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_mcq.py --task "pen-human-v1" --seed 99 --epoch 200 --step-per-epoch 250 --interaction_tag False & 



# maze2d-large-dense-v1
CUDA_VISIBLE_DEVICES=0 python run_example/run_cql.py --task "maze2d-large-dense-v1" --seed 33 --epoch 200 --step-per-epoch 250 --interaction_tag True & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_iql.py --task "maze2d-large-dense-v1" --seed 33 --epoch 200 --step-per-epoch 250 --interaction_tag True & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_td3bc.py --task "maze2d-large-dense-v1" --seed 33 --epoch 200 --step-per-epoch 250 --interaction_tag True & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_mcq.py --task "maze2d-large-dense-v1" --seed 33 --epoch 200 --step-per-epoch 250 --interaction_tag True & 

CUDA_VISIBLE_DEVICES=0 python run_example/run_cql.py --task "maze2d-large-dense-v1" --seed 66 --epoch 200 --step-per-epoch 250 --interaction_tag True & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_iql.py --task "maze2d-large-dense-v1" --seed 66 --epoch 200 --step-per-epoch 250 --interaction_tag True & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_td3bc.py --task "maze2d-large-dense-v1" --seed 66 --epoch 200 --step-per-epoch 250 --interaction_tag True & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_mcq.py --task "maze2d-large-dense-v1" --seed 66 --epoch 200 --step-per-epoch 250 --interaction_tag True & 

CUDA_VISIBLE_DEVICES=0 python run_example/run_cql.py --task "maze2d-large-dense-v1" --seed 99 --epoch 200 --step-per-epoch 250 --interaction_tag True & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_iql.py --task "maze2d-large-dense-v1" --seed 99 --epoch 200 --step-per-epoch 250 --interaction_tag True & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_td3bc.py --task "maze2d-large-dense-v1" --seed 99 --epoch 200 --step-per-epoch 250 --interaction_tag True & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_mcq.py --task "maze2d-large-dense-v1" --seed 99 --epoch 200 --step-per-epoch 250 --interaction_tag True & 


CUDA_VISIBLE_DEVICES=1 python run_example/run_cql.py --task "maze2d-large-dense-v1" --seed 33 --epoch 200 --step-per-epoch 250 --interaction_tag False & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_iql.py --task "maze2d-large-dense-v1" --seed 33 --epoch 200 --step-per-epoch 250 --interaction_tag False & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_td3bc.py --task "maze2d-large-dense-v1" --seed 33 --epoch 200 --step-per-epoch 250 --interaction_tag False & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_mcq.py --task "maze2d-large-dense-v1" --seed 33 --epoch 200 --step-per-epoch 250 --interaction_tag False & 

CUDA_VISIBLE_DEVICES=1 python run_example/run_cql.py --task "maze2d-large-dense-v1" --seed 66 --epoch 200 --step-per-epoch 250 --interaction_tag False & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_iql.py --task "maze2d-large-dense-v1" --seed 66 --epoch 200 --step-per-epoch 250 --interaction_tag False & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_td3bc.py --task "maze2d-large-dense-v1" --seed 66 --epoch 200 --step-per-epoch 250 --interaction_tag False & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_mcq.py --task "maze2d-large-dense-v1" --seed 66 --epoch 200 --step-per-epoch 250 --interaction_tag False & 

CUDA_VISIBLE_DEVICES=1 python run_example/run_cql.py --task "maze2d-large-dense-v1" --seed 99 --epoch 200 --step-per-epoch 250 --interaction_tag False & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_iql.py --task "maze2d-large-dense-v1" --seed 99 --epoch 200 --step-per-epoch 250 --interaction_tag False & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_td3bc.py --task "maze2d-large-dense-v1" --seed 99 --epoch 200 --step-per-epoch 250 --interaction_tag False & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_mcq.py --task "maze2d-large-dense-v1" --seed 99 --epoch 200 --step-per-epoch 250 --interaction_tag False & 



# hopper_random-v2
CUDA_VISIBLE_DEVICES=0 python run_example/run_cql.py --task "hopper_random-v2" --seed 33 --epoch 200 --step-per-epoch 250 --interaction_tag True & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_iql.py --task "hopper_random-v2" --seed 33 --epoch 200 --step-per-epoch 250 --interaction_tag True & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_td3bc.py --task "hopper_random-v2" --seed 33 --epoch 200 --step-per-epoch 250 --interaction_tag True & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_mcq.py --task "hopper_random-v2" --seed 33 --epoch 200 --step-per-epoch 250 --interaction_tag True & 

CUDA_VISIBLE_DEVICES=0 python run_example/run_cql.py --task "hopper_random-v2" --seed 66 --epoch 200 --step-per-epoch 250 --interaction_tag True & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_iql.py --task "hopper_random-v2" --seed 66 --epoch 200 --step-per-epoch 250 --interaction_tag True & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_td3bc.py --task "hopper_random-v2" --seed 66 --epoch 200 --step-per-epoch 250 --interaction_tag True & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_mcq.py --task "hopper_random-v2" --seed 66 --epoch 200 --step-per-epoch 250 --interaction_tag True & 

CUDA_VISIBLE_DEVICES=0 python run_example/run_cql.py --task "hopper_random-v2" --seed 99 --epoch 200 --step-per-epoch 250 --interaction_tag True & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_iql.py --task "hopper_random-v2" --seed 99 --epoch 200 --step-per-epoch 250 --interaction_tag True & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_td3bc.py --task "hopper_random-v2" --seed 99 --epoch 200 --step-per-epoch 250 --interaction_tag True & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_mcq.py --task "hopper_random-v2" --seed 99 --epoch 200 --step-per-epoch 250 --interaction_tag True & 


CUDA_VISIBLE_DEVICES=1 python run_example/run_cql.py --task "hopper_random-v2" --seed 33 --epoch 200 --step-per-epoch 250 --interaction_tag False & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_iql.py --task "hopper_random-v2" --seed 33 --epoch 200 --step-per-epoch 250 --interaction_tag False & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_td3bc.py --task "hopper_random-v2" --seed 33 --epoch 200 --step-per-epoch 250 --interaction_tag False & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_mcq.py --task "hopper_random-v2" --seed 33 --epoch 200 --step-per-epoch 250 --interaction_tag False & 

CUDA_VISIBLE_DEVICES=1 python run_example/run_cql.py --task "hopper_random-v2" --seed 66 --epoch 200 --step-per-epoch 250 --interaction_tag False & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_iql.py --task "hopper_random-v2" --seed 66 --epoch 200 --step-per-epoch 250 --interaction_tag False & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_td3bc.py --task "hopper_random-v2" --seed 66 --epoch 200 --step-per-epoch 250 --interaction_tag False & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_mcq.py --task "hopper_random-v2" --seed 66 --epoch 200 --step-per-epoch 250 --interaction_tag False & 

CUDA_VISIBLE_DEVICES=1 python run_example/run_cql.py --task "hopper_random-v2" --seed 99 --epoch 200 --step-per-epoch 250 --interaction_tag False & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_iql.py --task "hopper_random-v2" --seed 99 --epoch 200 --step-per-epoch 250 --interaction_tag False & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_td3bc.py --task "hopper_random-v2" --seed 99 --epoch 200 --step-per-epoch 250 --interaction_tag False & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_mcq.py --task "hopper_random-v2" --seed 99 --epoch 200 --step-per-epoch 250 --interaction_tag False & 



# hopper_medium_expert-v2
CUDA_VISIBLE_DEVICES=0 python run_example/run_cql.py --task "hopper_medium_expert-v2" --seed 33 --epoch 200 --step-per-epoch 250 --interaction_tag True & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_iql.py --task "hopper_medium_expert-v2" --seed 33 --epoch 200 --step-per-epoch 250 --interaction_tag True & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_td3bc.py --task "hopper_medium_expert-v2" --seed 33 --epoch 200 --step-per-epoch 250 --interaction_tag True & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_mcq.py --task "hopper_medium_expert-v2" --seed 33 --epoch 200 --step-per-epoch 250 --interaction_tag True & 

CUDA_VISIBLE_DEVICES=0 python run_example/run_cql.py --task "hopper_medium_expert-v2" --seed 66 --epoch 200 --step-per-epoch 250 --interaction_tag True & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_iql.py --task "hopper_medium_expert-v2" --seed 66 --epoch 200 --step-per-epoch 250 --interaction_tag True & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_td3bc.py --task "hopper_medium_expert-v2" --seed 66 --epoch 200 --step-per-epoch 250 --interaction_tag True & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_mcq.py --task "hopper_medium_expert-v2" --seed 66 --epoch 200 --step-per-epoch 250 --interaction_tag True & 

CUDA_VISIBLE_DEVICES=0 python run_example/run_cql.py --task "hopper_medium_expert-v2" --seed 99 --epoch 200 --step-per-epoch 250 --interaction_tag True & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_iql.py --task "hopper_medium_expert-v2" --seed 99 --epoch 200 --step-per-epoch 250 --interaction_tag True & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_td3bc.py --task "hopper_medium_expert-v2" --seed 99 --epoch 200 --step-per-epoch 250 --interaction_tag True & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_mcq.py --task "hopper_medium_expert-v2" --seed 99 --epoch 200 --step-per-epoch 250 --interaction_tag True & 


CUDA_VISIBLE_DEVICES=1 python run_example/run_cql.py --task "hopper_medium_expert-v2" --seed 33 --epoch 200 --step-per-epoch 250 --interaction_tag False & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_iql.py --task "hopper_medium_expert-v2" --seed 33 --epoch 200 --step-per-epoch 250 --interaction_tag False & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_td3bc.py --task "hopper_medium_expert-v2" --seed 33 --epoch 200 --step-per-epoch 250 --interaction_tag False & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_mcq.py --task "hopper_medium_expert-v2" --seed 33 --epoch 200 --step-per-epoch 250 --interaction_tag False & 

CUDA_VISIBLE_DEVICES=1 python run_example/run_cql.py --task "hopper_medium_expert-v2" --seed 66 --epoch 200 --step-per-epoch 250 --interaction_tag False & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_iql.py --task "hopper_medium_expert-v2" --seed 66 --epoch 200 --step-per-epoch 250 --interaction_tag False & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_td3bc.py --task "hopper_medium_expert-v2" --seed 66 --epoch 200 --step-per-epoch 250 --interaction_tag False & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_mcq.py --task "hopper_medium_expert-v2" --seed 66 --epoch 200 --step-per-epoch 250 --interaction_tag False & 

CUDA_VISIBLE_DEVICES=1 python run_example/run_cql.py --task "hopper_medium_expert-v2" --seed 99 --epoch 200 --step-per-epoch 250 --interaction_tag False & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_iql.py --task "hopper_medium_expert-v2" --seed 99 --epoch 200 --step-per-epoch 250 --interaction_tag False & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_td3bc.py --task "hopper_medium_expert-v2" --seed 99 --epoch 200 --step-per-epoch 250 --interaction_tag False & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_mcq.py --task "hopper_medium_expert-v2" --seed 99 --epoch 200 --step-per-epoch 250 --interaction_tag False & 



# halfcheetah_medium_expert-v2
CUDA_VISIBLE_DEVICES=0 python run_example/run_cql.py --task "halfcheetah_medium_expert-v2" --seed 33 --epoch 200 --step-per-epoch 250 --interaction_tag True & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_iql.py --task "halfcheetah_medium_expert-v2" --seed 33 --epoch 200 --step-per-epoch 250 --interaction_tag True & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_td3bc.py --task "halfcheetah_medium_expert-v2" --seed 33 --epoch 200 --step-per-epoch 250 --interaction_tag True & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_mcq.py --task "halfcheetah_medium_expert-v2" --seed 33 --epoch 200 --step-per-epoch 250 --interaction_tag True & 

CUDA_VISIBLE_DEVICES=0 python run_example/run_cql.py --task "halfcheetah_medium_expert-v2" --seed 66 --epoch 200 --step-per-epoch 250 --interaction_tag True & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_iql.py --task "halfcheetah_medium_expert-v2" --seed 66 --epoch 200 --step-per-epoch 250 --interaction_tag True & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_td3bc.py --task "halfcheetah_medium_expert-v2" --seed 66 --epoch 200 --step-per-epoch 250 --interaction_tag True & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_mcq.py --task "halfcheetah_medium_expert-v2" --seed 66 --epoch 200 --step-per-epoch 250 --interaction_tag True & 

CUDA_VISIBLE_DEVICES=0 python run_example/run_cql.py --task "halfcheetah_medium_expert-v2" --seed 99 --epoch 200 --step-per-epoch 250 --interaction_tag True & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_iql.py --task "halfcheetah_medium_expert-v2" --seed 99 --epoch 200 --step-per-epoch 250 --interaction_tag True & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_td3bc.py --task "halfcheetah_medium_expert-v2" --seed 99 --epoch 200 --step-per-epoch 250 --interaction_tag True & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_mcq.py --task "halfcheetah_medium_expert-v2" --seed 99 --epoch 200 --step-per-epoch 250 --interaction_tag True & 


CUDA_VISIBLE_DEVICES=1 python run_example/run_cql.py --task "halfcheetah_medium_expert-v2" --seed 33 --epoch 200 --step-per-epoch 250 --interaction_tag False & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_iql.py --task "halfcheetah_medium_expert-v2" --seed 33 --epoch 200 --step-per-epoch 250 --interaction_tag False & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_td3bc.py --task "halfcheetah_medium_expert-v2" --seed 33 --epoch 200 --step-per-epoch 250 --interaction_tag False & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_mcq.py --task "halfcheetah_medium_expert-v2" --seed 33 --epoch 200 --step-per-epoch 250 --interaction_tag False & 

CUDA_VISIBLE_DEVICES=1 python run_example/run_cql.py --task "halfcheetah_medium_expert-v2" --seed 66 --epoch 200 --step-per-epoch 250 --interaction_tag False & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_iql.py --task "halfcheetah_medium_expert-v2" --seed 66 --epoch 200 --step-per-epoch 250 --interaction_tag False & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_td3bc.py --task "halfcheetah_medium_expert-v2" --seed 66 --epoch 200 --step-per-epoch 250 --interaction_tag False & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_mcq.py --task "halfcheetah_medium_expert-v2" --seed 66 --epoch 200 --step-per-epoch 250 --interaction_tag False & 

CUDA_VISIBLE_DEVICES=1 python run_example/run_cql.py --task "halfcheetah_medium_expert-v2" --seed 99 --epoch 200 --step-per-epoch 250 --interaction_tag False & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_iql.py --task "halfcheetah_medium_expert-v2" --seed 99 --epoch 200 --step-per-epoch 250 --interaction_tag False & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_td3bc.py --task "halfcheetah_medium_expert-v2" --seed 99 --epoch 200 --step-per-epoch 250 --interaction_tag False & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_mcq.py --task "halfcheetah_medium_expert-v2" --seed 99 --epoch 200 --step-per-epoch 250 --interaction_tag False & 



# halfcheetah_random-v2
CUDA_VISIBLE_DEVICES=0 python run_example/run_cql.py --task "halfcheetah_random-v2" --seed 33 --epoch 200 --step-per-epoch 250 --interaction_tag True & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_iql.py --task "halfcheetah_random-v2" --seed 33 --epoch 200 --step-per-epoch 250 --interaction_tag True & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_td3bc.py --task "halfcheetah_random-v2" --seed 33 --epoch 200 --step-per-epoch 250 --interaction_tag True & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_mcq.py --task "halfcheetah_random-v2" --seed 33 --epoch 200 --step-per-epoch 250 --interaction_tag True & 

CUDA_VISIBLE_DEVICES=0 python run_example/run_cql.py --task "halfcheetah_random-v2" --seed 66 --epoch 200 --step-per-epoch 250 --interaction_tag True & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_iql.py --task "halfcheetah_random-v2" --seed 66 --epoch 200 --step-per-epoch 250 --interaction_tag True & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_td3bc.py --task "halfcheetah_random-v2" --seed 66 --epoch 200 --step-per-epoch 250 --interaction_tag True & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_mcq.py --task "halfcheetah_random-v2" --seed 66 --epoch 200 --step-per-epoch 250 --interaction_tag True & 

CUDA_VISIBLE_DEVICES=0 python run_example/run_cql.py --task "halfcheetah_random-v2" --seed 99 --epoch 200 --step-per-epoch 250 --interaction_tag True & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_iql.py --task "halfcheetah_random-v2" --seed 99 --epoch 200 --step-per-epoch 250 --interaction_tag True & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_td3bc.py --task "halfcheetah_random-v2" --seed 99 --epoch 200 --step-per-epoch 250 --interaction_tag True & 
CUDA_VISIBLE_DEVICES=0 python run_example/run_mcq.py --task "halfcheetah_random-v2" --seed 99 --epoch 200 --step-per-epoch 250 --interaction_tag True & 


CUDA_VISIBLE_DEVICES=1 python run_example/run_cql.py --task "halfcheetah_random-v2" --seed 33 --epoch 200 --step-per-epoch 250 --interaction_tag False & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_iql.py --task "halfcheetah_random-v2" --seed 33 --epoch 200 --step-per-epoch 250 --interaction_tag False & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_td3bc.py --task "halfcheetah_random-v2" --seed 33 --epoch 200 --step-per-epoch 250 --interaction_tag False & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_mcq.py --task "halfcheetah_random-v2" --seed 33 --epoch 200 --step-per-epoch 250 --interaction_tag False & 

CUDA_VISIBLE_DEVICES=1 python run_example/run_cql.py --task "halfcheetah_random-v2" --seed 66 --epoch 200 --step-per-epoch 250 --interaction_tag False & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_iql.py --task "halfcheetah_random-v2" --seed 66 --epoch 200 --step-per-epoch 250 --interaction_tag False & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_td3bc.py --task "halfcheetah_random-v2" --seed 66 --epoch 200 --step-per-epoch 250 --interaction_tag False & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_mcq.py --task "halfcheetah_random-v2" --seed 66 --epoch 200 --step-per-epoch 250 --interaction_tag False & 

CUDA_VISIBLE_DEVICES=1 python run_example/run_cql.py --task "halfcheetah_random-v2" --seed 99 --epoch 200 --step-per-epoch 250 --interaction_tag False & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_iql.py --task "halfcheetah_random-v2" --seed 99 --epoch 200 --step-per-epoch 250 --interaction_tag False & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_td3bc.py --task "halfcheetah_random-v2" --seed 99 --epoch 200 --step-per-epoch 250 --interaction_tag False & 
CUDA_VISIBLE_DEVICES=1 python run_example/run_mcq.py --task "halfcheetah_random-v2" --seed 99 --epoch 200 --step-per-epoch 250 --interaction_tag False & 