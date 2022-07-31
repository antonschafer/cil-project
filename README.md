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

## Run baselines:
- To run the TFIDF baseline run python3 baselines.py --tfidf --full_data
- To run the Glove baseline run python3 baselines.py --glove --full_data

## GPT-J Embeddings:
- Requires a GPU with more than 12GB of memory.
- Run python -m generate_embeddings  --model gptj --save_to_wandb


## Run ensemble:
1. Make sure all runs you want to use as submodels have train_ensemble_preds.npy, val_preds.npy, val_final_preds.npy, and test_preds.npy files. If they don't exist for a particular run with a model, run `python -m test --model ... --run_id ...  --save_to_wandb` to generate them.
2. Include embeddings by including a run where embeddings were predicted (e.g. run 3tc5cxtj or 35hiik2s)
3. Get the run ids of all runs you want to use and run via e.g.: `python -m train_ensemble --nepochs 50 --dropout 0.2 --hidden_size 512 --batch_size 64 --lr 0.001 --val_check_interval 1.0 --model_runs 3tc5cxtj 35hiik2s 37brzj6x 1rmln14m 3kr9nxd1  --es_patience 5 --run_name ensemble-example`
