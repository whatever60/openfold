#!/bin/bash

#SBATCH -J download_esmfold_atlas           # Job name
#SBATCH -o slurm_log/download_esmfold_atlas.o%j       # Name of stdout output file
#SBATCH -e slurm_log/download_esmfold_atlas.e%j       # Name of stderr error file
#SBATCH -p development
#SBATCH -n 1
#SBATCH -t 48:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH --mail-user=yq2355@cumc.columbia.edu

srun scripts_custom/download_esmfold_atlas.sh
