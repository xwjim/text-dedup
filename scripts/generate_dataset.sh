#!/bin/bash

# 转化为lmflow格式
python data_preprocess/lmflow_format.py --dataset_path output/minhash/BELLE_dedup/ --output_path output/minhash/BELLE_dedup/lm_train.json --option hf2lm --language chinese
python data_preprocess/lmflow_format.py --dataset_path output/minhash/dolly_dedup/ --output_path output/minhash/dolly_dedup/lm_train.json --option hf2lm --language english
python data_preprocess/lmflow_format.py --dataset_path output/minhash/gpt4all-clean_dedup/ --output_path output/minhash/gpt4all-clean_dedup/lm_train.json --option hf2lm --language english
python data_preprocess/lmflow_format.py --dataset_path output/minhash/unnatural-instructions_dedup/ --output_path output/minhash/unnatural-instructions_dedup/lm_train.json --option hf2lm --language english

# 合并
python data_preprocess/merge.py --dataset_path output/minhash/gpt4all-clean_dedup/lm_train.json --merge_from_path output/minhash/unnatural-instructions_dedup/lm_train.json --output_path dataset/en_mergy/lm_train.json
python data_preprocess/shuffle.py --dataset_path dataset/en_mergy/lm_train.json --output_path dataset/en_mergy/lm_train.json
python data_preprocess/sample.py --dataset_path dataset/en_mergy/lm_train.json --output_path dataset/en_mergy/lm_train.json --ratio 0.23

python data_preprocess/merge.py --dataset_path dataset/en_mergy/lm_train.json --merge_from_path output/minhash/dolly_dedup/lm_train.json --output_path dataset/en_mergy/lm_train.json

#产生中文
python data_preprocess/sample.py --dataset_path output/minhash/BELLE_dedup/lm_train.json --output_path dataset/en_mergy/lm_train_ch.json --ratio 0.217
python data_preprocess/merge.py --dataset_path dataset/en_mergy/lm_train.json --merge_from_path dataset/en_mergy/lm_train_ch.json --output_path dataset/en_mergy/lm_train.json

python data_preprocess/shuffle.py --dataset_path dataset/en_mergy/lm_train.json --output_path dataset/en_mergy/lm_train.json

python data_preprocess/count.py --dataset_path dataset/en_mergy/lm_train.json