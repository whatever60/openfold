#!/bin/bash

source ~/.bashrc
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
# conda activate /scratch/00946/zzhang/python-env/openfold-venv
source scripts/activate_conda_env.sh
export CUDA_HOME=/opt/apps/cuda/11.4

# pip3 install rich -q

sample_data_dir=/scratch/09101/whatever/data/openfold_sample

echo running full $(date)

python train_openfold.py \
     $sample_data_dir/mmcif_files \
     $sample_data_dir/alignment_db \
     $sample_data_dir/mmcif_files \
     train_baseline/sample \
     2021-10-01 \
     --alignment_index_path $sample_data_dir/alignment_db/dup_super.index \
     --val_data_dir $sample_data_dir/val/mmcif_files \
     --val_alignment_dir $sample_data_dir/val/alignments \
     --template_release_dates_cache_path $sample_data_dir/mmcif_cache.json \
     --train_epoch_len 50 \
     --accumulate_grad_batches 3 \
     --replace_sampler_ddp True \
     --log_lr \
     --deepspeed_config_path deepspeed_config.json \
     --checkpoint_every_epoch \
     --obsolete_pdbs_file_path $sample_data_dir/obsolete.dat \
     --train_chain_data_cache_path $sample_data_dir/prot_data_cache.json \
     --seed $(date +"%m%d%Y") \
     --config_preset initial_training \
     --gpus $1 \
     --num_nodes $2 \
     # --wandb \
     # --wandb_entity openfold \
     # --wandb_project single_sequence_yiming  \
     # --experiment_name full_replicate_zhao \

     # --rich
     # --resume_from_ckpt train_gustaf_output/baseline/openfold-ls6/v8be17mz/checkpoints/4-4999.ckpt/

#      --alignment_index_path=/scratch/00946/zzhang/data/openfold/ls6-tacc/alignment_db/dup_super.index \
#     --wandb \
#     --wandb_project openfold \
#     --wandb_entity zhaozhang \
#     --experiment_name test-4nodes-ls6 \
#     2>&1 | tee openfold-4node-1.log

#      --resume_from_ckpt full_output/openfold/a858ayiy/checkpoints/2-2999.ckpt/
