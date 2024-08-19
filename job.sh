#!/bin/sh
#SBATCH --job-name="julia_test"  # Job name
#SBATCH --output=julia_out       # Standard output file
#SBATCH --error=julia_err        # Standard error file
#SBATCH --mail-type=END          # Email notification type
#SBATCH --mail-user=jimmyli2026@u.northwestern.edu  # Email address for notifications

#SBATCH -N 1  # Number of nodes
#SBATCH -n 1  # Number of tasks
#SBATCH --cpus-per-task=4  # Number of CPU cores per task

#SBATCH --mem=16G  # Total memory per node

module load julia/10.1  # Load the Julia module (adjust the version as needed)
module load ngspice/43

julia two_dimer_opt.jl

cp * ~/tmp/