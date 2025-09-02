#!/bin/bash

TIME=$(date "+%Y%m%d-%H%M%S")

nohup /ai/anaconda3/envs/gpt/bin/python train.py \
  --dataset_date 0710 \
  --model_name_or_path /ai/users/dxd/notebook/pretrained/trained_model/attribute_model/checkpoint-1425/ \
  --token_label_file /ai/users/dxd/notebook/notebooks/standard_material/huaneng/attribute_extraction/gathered_label2id.json \
  --train_file /ai/users/dxd/notebook/notebooks/standard_material/huaneng/attribute_extraction/merged_train.txt \
  --validation_file /ai/users/dxd/notebook/notebooks/standard_material/huaneng/attribute_extraction/merged_dev.txt \
  --test_file /ai/users/dxd/notebook/notebooks/standard_material/huaneng/attribute_extraction/merged_test.txt \
  --overwrite_cache True \
  --output_dir output \
  --num_train_epochs 1 \
  --per_device_train_batch_size 64 \
  --per_device_eval_batch_size 64 \
  --learning_rate 2e-5 \
  --max_seq_length 128 \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --logging_strategy steps \
  --logging_steps 100 \
  --load_best_model_at_end True \
  --do_train \
  --do_eval >logs/model_train_"${TIME}".log 2>&1 &
