#!/bin/bash
# entrypoint.sh: Setup environment and execute command
set -e # Exit immediately if a command exits with a non-zero status.

echo "Activating A-Sys-I conda environment..."
# Source conda script to make 'conda' command available
source /root/miniconda3/etc/profile.d/conda.sh 
conda activate asys-i

echo "Environment activated. Running command: $@"
# Execute the command passed to docker run
exec "$@" 
