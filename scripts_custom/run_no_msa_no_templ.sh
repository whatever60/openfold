#!/bin/bash

source ~/.bashrc
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
# conda activate /scratch/00946/zzhang/python-env/openfold-venv
source scripts/activate_conda_env.sh
export CUDA_HOME=/opt/apps/cuda/11.4

# pip3 install rich -q

echo running no msa no template baseline $(date)

echo $(nvidia-smi)

python train_openfold.py \
     /scratch/00946/zzhang/data/openfold/ls6-tacc/pdb_mmcif/mmcif_files \
     /scratch/09101/whatever/data/openfold/alignment_db \
     /scratch/00946/zzhang/data/openfold/ls6-tacc/pdb_mmcif/mmcif_files \
     train_baseline/test \
     2021-10-01 \
     --alignment_index_path /scratch/09101/whatever/data/openfold/alignment_db/duplicated_super_fix.index \
     --val_data_dir /scratch/00946/zzhang/data/openfold/ls6-tacc/gustaf/val_set/data \
     --val_alignment_dir /scratch/09101/whatever/data/openfold/val_set \
     --template_release_dates_cache_path /scratch/00946/zzhang/data/openfold/ls6-tacc/gustaf/mmcif_cache.json \
     --train_epoch_len 126000 \
     --accumulate_grad_batches 3 \
     --replace_sampler_ddp True \
     --log_lr \
     --deepspeed_config_path deepspeed_config.json \
     --checkpoint_every_epoch \
     --obsolete_pdbs_file_path /scratch/00946/zzhang/data/openfold/ls6-tacc/pdb_mmcif/obsolete.dat \
     --train_chain_data_cache_path /scratch/00946/zzhang/data/openfold/ls6-tacc/gustaf/prot_data_cache.json \
     --wandb \
     --wandb_entity openfold \
     --wandb_project single_sequence_yiming  \
     --experiment_name no_msa_no_template \
     --seed $(date +"%m%d%Y") \
     --config_preset no_msa_no_template \
     --gpus $1 \
     --num_nodes $2 \
     --resume_from_ckpt train_baseline/test/single_sequence_yiming/tn1tdqyg/checkpoints/52-185499.ckpt/
     # --rich
     # --resume_from_ckpt train_gustaf_output/baseline/openfold-ls6/v8be17mz/checkpoints/4-4999.ckpt/

#      --alignment_index_path=/scratch/00946/zzhang/data/openfold/ls6-tacc/alignment_db/duplicated_super.index \
#     --wandb \
#     --wandb_project openfold \
#     --wandb_entity zhaozhang \
#     --experiment_name test-4nodes-ls6 \
#     2>&1 | tee openfold-4node-1.log

#      --resume_from_ckpt full_output/openfold/a858ayiy/checkpoints/2-2999.ckpt/
