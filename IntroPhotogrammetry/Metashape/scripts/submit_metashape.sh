#!/bin/bash

# job standard output will go to the file slurm-%j.out (where %j is the job ID)
# DEFINE SLURM VARIABLES
#SBATCH --job-name="metashape"
#SBATCH --partition=atlas                 # GPU node(s): 'atlas' or 'gpu' partition
#SBATCH --nodes=1                         # number of nodes
#SBATCH --ntasks=48                       # 24 processor core(s) per node X 2 threads per core
#SBATCH --time=01:30:00                   # walltime limit (HH:MM:SS)
#SBATCH --account=isu_gif_vrsc
#SBATCH --mail-user=your.email@usda.gov   # email address
#SBATCH --mail-type=BEGIN                 # email notice of job started
#SBATCH --mail-type=END                   # email notice of job finished
#SBATCH --mail-type=FAIL                  # email notice of job failure


# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module load gcc/10.2.0                    # load gcc dependency
module load metashape                     # load metashape, then run script with x11 turned off

# DEFINE CODE VARIABLES
script_dir=/project/90daydata/isu_gif_vrsc/agisoft      # path to scripts in your workdir (can check with 'pwd' in terminal)
script_name=01_metashape_SPC.py                         # the filename of the python script you want to run

# DEFINE METASHAPE COMMAND
metashape -r $script_dir/$script_name -platform offscreen
