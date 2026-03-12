#!/bin/bash
#SBATCH --job-name=listops_grokking
#SBATCH --time=01:00:00           
#SBATCH --nodes=1                 
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1         
#SBATCH --cpus-per-task=16        
#SBATCH --output=benchmark_%j.log

echo "Sending job to GH200 compute node..."

# Execute everything inside the specific PyTorch image on the GPU node
srun --uenv=pytorch/v2.9.1:v2 --view=default bash -c '
    echo "Configuring Python environment..."
    unset PYTHONPATH
    export PYTHONUSERBASE="$(dirname "$(dirname "$(which python)")")"
    
    python -m venv --system-site-packages my_env
    source my_env/bin/activate
    
    echo "Installing requirements..."
    pip install -r requirements.txt
    export TORCHINDUCTOR_CACHE_DIR="/capstor/scratch/cscs/$USER/torch_cache"
 
    echo "Executing the Ultimate Benchmark Suite..."
    python -u run_experiments.py
'
