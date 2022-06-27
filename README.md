# cil-project

## Run on euler:
1. install miniconda
2. create environment and install pip
    - conda create -n cil
    - conda activate cil
    - conda install pip
3. install packages (pip install -r requirements.txt)
4. clone repos if necessary (using git lfs if necesssary)

## To run using weights and biases
1. Make sure wandb is installed (pip install -r requirements.txt)
2. Login to wandb: wandb login
4. After experiments have run, sync via: wandb sync wandb/offline-run*