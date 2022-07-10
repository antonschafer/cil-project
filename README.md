# cil-project

## Setup on euler:
1. install miniconda
2. create environment and install pip
    - `conda create -n cil`
    - `conda activate cil`
    - `conda install pip`
3. install packages `pip install -r requirements.txt`
4. Login to wandb: wandb login
5. Run experiments, e.g. `bsub -R "rusage[mem=32000,ngpus_excl_p=1]" -W 12:00 bash sample_job.sh`
    - for interactive job run e.g `bsub -R "rusage[mem=32000,ngpus_excl_p=1]" -Ip bash` (then for setup `source setup.sh`)