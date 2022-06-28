# activate environment on compute node
eval "$(conda shell.bash hook)"
conda activate cil

# run training, e.g.
python -m train --model twitter_roberta --nepochs 2 --full_data --run_name twitter_roberta_full_data
