#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model_name=TimesNet

    python -u ../run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --des train_run \
      --root_path ../data \
      --data_path train_timesnet.csv \
      --model_id cooling_demand_72_24 \
      --model $model_name \
      --checkpoints ../checkpoints/ \
      --data custom \
      --features M \
      --seq_len 168 \
      --label_len 24 \
      --pred_len 24 \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 37 \
      --dec_in 37 \
      --c_out 1 \
      --target cooling_demand \
      --d_model 128 \
      --d_ff 256 \
      --top_k 5 \
      --itr 1 \
      --n_heads 4 \
      --num_workers 6 \
      --batch_size 64 \
      --use_norm 1 \
      --patience 4 \
      --train_epochs 15
