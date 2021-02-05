#!/bin/bash

# TF_XLA_FLAGS=--tf_xla_auto_jit=2
python3 bigbird/summarization/run_summarization.py \
  --data_dir="/data/ysc/pycharm_work_space/LongSumm/bigbird/dataset" \
  --output_dir="/data/ysc/pycharm_work_space/LongSumm/bigbird/output_dir" \
  --attention_type=block_sparse \
  --couple_encoder_decoder=False \
  --max_encoder_length=4096 \
  --max_decoder_length=1024 \
  --num_attention_heads=16 \
  --num_hidden_layers=16 \
  --hidden_size=1024 \
  --intermediate_size=4096 \
  --block_size=64 \
  --scope=pegasus \
  --norm_type=prenorm \
  --hidden_act=relu \
  --use_bias=False \
  --rescale_embedding=True \
  --vocab_model_file=pegasus \
  --substitute_newline="<n>" \
  --train_batch_size=2 \
  --eval_batch_size=2 \
  --do_train=True \
  --do_eval=False \
  --use_tpu=False \
  --tpu_name=bigbird \
  --tpu_zone=europe-west4-a \
  --gcp_project="$GCP_PROJECT_NAME" \
  --num_tpu_cores=128 \
  --init_checkpoint=/data/ysc/pycharm_work_space/LongSumm/pretrain_model/bigbird_pegasus/model.ckpt-300000
