### Training on SATLIB graphs of the MIS problem

export PATH=/usr/local/cuda-11.3/bin:$PATH  
export LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64:$LD_LIBRARY_PATH

export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# shellcheck disable=SC2155
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
echo "WANDB_ID is $WANDB_RUN_ID"

python -u difusco/train.py \
  --task "mis" \
  --wandb_logger_name "mis_diffusion_graph_categorical_sat" \
  --diffusion_type "categorical" \
  --do_train \
  --do_test \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "storage" \
  --training_split "train_mis_sat/*gpickle" \
  --validation_split "test_mis_sat/*gpickle" \
  --test_split "test_mis_sat/*gpickle" \
  --batch_size 16 \
  --num_epochs 50 \
  --validation_examples 8 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 50 \
  --use_activation_checkpoint
