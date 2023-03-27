#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=07:59:59
#SBATCH --job-name=[NAME JOB]
#SBATCH --mem=32GB
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --output=myjob.%j.out
#SBATCH --error=myjob.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your_email@northeastern.edu


module load cuda/11.3
module load anaconda3/2022.01
source miniconda3/bin/activate
source activate astro

python -c'import torch; print(torch.cuda.is_available())'
python /home/pandya.sne/GCNNMorphology/src/scripts/train.py --config ../config/resnet50.yaml
