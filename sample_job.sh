# activate environment on compute node
eval "$(conda shell.bash hook)"
conda activate cil

# make network available
module load eth_proxy

# set huggingface cache directory to scratch where more space than in home
export HUGGINGFACE_CACHE_DIR=/cluster/scratch/$USER/huggingface_cache


# run training, e.g.
python -m train --model twitter_roberta --nepochs 3 --full_data --run_name sample_job --lr 5e-5 --batch_size 64 --accumulate_grad_batches 2 --seed 0 --train_data_size 1