#!/bin/bash

#SBATCH -J openfold           # Job name
#SBATCH -o slurm_log/openfold.o%j       # Name of stdout output file
#SBATCH -e slurm_log/openfold.e%j       # Name of stderr error file
#SBATCH -p gpu-a100          # Queue (partition) name
#SBATCH -N 2               # Total # of nodes
#SBATCH -n 6              # Total # of mpi tasks
#SBATCH -t 24:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH --mail-user=yq2355@cumc.columbia.edu

srun scripts_custom/run_full.sh 3 2
