# cil-project

## Setup on euler:
1. install miniconda
2. create environment and install pip
    - `conda create -n cil`
    - `conda activate cil`
    - `conda install pip`
3. install packages `pip install -r requirements.txt`
4. create datasplit `python create_splits.py`
5. Login to wandb: wandb login
6. Run experiments, e.g. `bsub -R "rusage[mem=48000,ngpus_excl_p=1]" -W 24:00 bash sample_job.sh`
    - for interactive job run e.g `bsub -R "rusage[mem=48000,ngpus_excl_p=1]" -Ip bash` (then for setup `source setup.sh`)


## Run ensemble:
1. Make sure all runs you want to use as submodels have val_preds.npy, val_final_preds.npy, and test_preds.npy files. If they don't exist for a particular run with a model, run `python -m test --model ... --run_id ...  --save_to_wandb` to generate them.
2. Include embeddings by including a run where embeddings were predicted (e.g. run 1oy7w4gu)
3. Get the run ids of all runs you want to use and run via: `python -m train_ensemble --nepochs ... --dropout ... --hidden_size ... --batch_size ... --run_name ... --model_runs ... ... ... ...`
