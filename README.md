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



## Base models
Distilbert emotion with whole training data: `python -m train --model distilbert_emotion --nepochs 3 --full_data --run_name distilbert_emotion --lr 2e-5 --batch_size 64 --seed 0 --train_size 1`

Bert with whole training data:  `python -m train --model base --nepochs 3 --full_data --run_name base --lr 2e-5 --batch_size 64 --seed 0 --train_size 1`

*Roberta with the whole training data: `python -m train --model twitter_roberta --nepochs 3 --full_data --run_name sample_job --lr 5e-5 --batch_size 64 --accumulate_grad_batches 2 --seed 0 --train_data_size 1`

*Bert with 25% of the training data: `python -m train --model base --nepochs 3 --full_data --run_name base --lr 2e-5 --batch_size 64 --seed 0 --train_data_size .25`

*Roberta with 25% of the training data: `python -m train --model twitter_roberta --nepochs 3 --full_data --run_name sample_job --lr 5e-5 --batch_size 64 --accumulate_grad_batches 2 --seed 3 --train_data_size .25`

Distilbert emotion with 25% of the training data: `python -m train --model distilbert_emotion --nepochs 3 --full_data --run_name distilbert_emotion --lr 2e-5 --batch_size 64 --seed 0 --train_size .25`

The seed changes the way the training data is split, so it is not the same 25% of the training data if the seed is different.

For each model, we tried three different seeds (Only when using 25% of the data). {0, 1, 2} for Bert, {3, 4, 5} for Roberta and {6, 7, 8} for Distilbert emotion.

Models with an asterix were the ones we used for the final ensemble.

## Compute Coverages:

`python compute_coverage.py --model_runs {MODEL ID'S YOU WANT TO COMBINE}`

Example: `python compute_coverage.py  --model_runs 1poxasnf 3hbr9c7b 2pd8pjdr z839cmze 1okolsk9`

The code will produce the coverage for all the possible combinations of UP to three models

## Run ensemble:

1. Make sure all runs you want to use as submodels have train_ensemble_preds.npy, val_preds.npy, val_final_preds.npy, and test_preds.npy files. If they don't exist for a particular run with a model, run `python -m test --model ... --run_id ...  --save_to_wandb` to generate them.
2. Include embeddings by including a run where embeddings were predicted (e.g. run 3tc5cxtj or 35hiik2s)
3. Get the run ids of all runs you want to use and run via e.g.: `python -m train_ensemble --nepochs 50 --dropout 0.2 --hidden_size 512 --batch_size 64 --lr 0.001 --val_check_interval 1.0 --model_runs 3tc5cxtj 35hiik2s 37brzj6x 1rmln14m 3kr9nxd1  --es_patience 5 --run_name ensemble-example`

Our final run: `python -m train_ensemble --nepochs 50 --dropout 0.2 --hidden_size 512 --batch_size 64 --lr 0.001 --val_check_interval 1.0 --model_runs 3hbr9c7b 2xowegtp 1agkremo --es_patience 5 --run_name ensemble-final`

The model runs correspond to the ID's of the 3 Base final models seen in the previous section

## Use GPT3:

1. Run `python gpt3/uncertain_predictions.py --cost_lim {HOW MUCH YOU ARE WILLING TO SPEND TO TRAIN GPT3} --run_id {RUN ID}`
2. Run `python gpt3/gpt3_pred.py --predict False  --run_id {RUN ID} --mask {NAME OF MASK GENERATED IN STEP 1}`
3. Run `openai tools fine_tunes.prepare_data -f {GENERATED CSV IN STEP 2}`
4. Run `openai api fine_tunes.create -t "openai-parsed_prepared_train.jsonl"\
    -v "openai-parsed_prepared_valid.jsonl"\
    --compute_classification_metrics\
    --classification_n_classes 2`
5. Run `python gpt3/gpt3_pred.py --predict True --run_id {RUN ID} --mask {NAME OF MASK GENERATED IN STEP 1}`

GPT3 did not improve our final results and was therefore not part of our final model.