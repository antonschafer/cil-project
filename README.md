# cil-project

## Setup on euler:
1. install miniconda
2. create environment and install pip
    - `conda create -n cil`
    - `conda activate cil`
    - `conda install pip`
3. install packages `pip install -r requirements.txt`
4. clone repos if necessary (using git lfs if necesssary, see description in utils)
5. Login to wandb: wandb login
6. Run experiments, e.g. `bsub -R "rusage[mem=32000,ngpus_excl_p=1]" -W 12:00 bash sample_job.sh`
    - for interactive job run e.g `bsub -R "rusage[mem=32000,ngpus_excl_p=1]" -Ip bash`
7. After experiments have run, sync to weights and biases via: `wandb sync /cluster/scratch/$USER/wandb/offline-run*` (or sync individual run, just make sure not sync debug/useless runs)