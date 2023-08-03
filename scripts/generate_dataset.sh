#!/bin/bash

# 转化为lmflow格式
python data_preprocess/format_changer.py --dataset_path output/minhash/BELLE_dedup/ --output_path output/minhash/BELLE_dedup/lm_train.json --option hf2lm --language chinese --instruction_key instruction --input_key input --output_key output
python data_preprocess/format_changer.py --dataset_path output/minhash/dolly_dedup/ --output_path output/minhash/dolly_dedup/lm_train.json --option hf2lm --language english --instruction_key instruction --input_key context --output_key response
python data_preprocess/format_changer.py --dataset_path output/minhash/gpt4all-clean_dedup/ --output_path output/minhash/gpt4all-clean_dedup/lm_train.json --option hf2lm --language english --instruction_key prompt --input_key error --output_key response
# python data_preprocess/format_changer.py --dataset_path output/minhash/unnatural-instructions_dedup/ --output_path output/minhash/unnatural-instructions_dedup/lm_train.json --option hf2lm --language english --instruction_key instruction_with_input --input_key "source" --output_key output

# 添加prompt
python data_preprocess/add_prompt.py --dataset_path output/minhash/BELLE_dedup/lm_train.json --output_path output/minhash/BELLE_dedup/lm_train.json --language chinese
python data_preprocess/add_prompt.py --dataset_path output/minhash/dolly_dedup/lm_train.json --output_path output/minhash/dolly_dedup/lm_train.json --language english
python data_preprocess/add_prompt.py --dataset_path output/minhash/gpt4all-clean_dedup/lm_train.json --output_path output/minhash/gpt4all-clean_dedup/lm_train.json --language english
# python data_preprocess/add_prompt.py --dataset_path output/minhash/dolly_dedup/lm_train.json --output_path output/minhash/unnatural-instructions_dedup/lm_train.json --language english

# 合并
# python data_preprocess/merge.py --dataset_path output/minhash/gpt4all-clean_dedup/lm_train.json --merge_from_path output/minhash/unnatural-instructions_dedup/lm_train.json --output_path dataset/en_mergy/lm_train.json
# python data_preprocess/shuffle.py --dataset_path dataset/en_mergy/lm_train.json --output_path dataset/en_mergy/lm_train.json
python data_preprocess/sample.py --dataset_path output/minhash/gpt4all-clean_dedup/lm_train.json --output_path dataset/en_mergy/lm_train.json --ratio 0.598
python data_preprocess/merge.py --dataset_path dataset/en_mergy/lm_train.json --merge_from_path output/minhash/dolly_dedup/lm_train.json --output_path dataset/en_mergy/lm_train.json

#产生中文
python data_preprocess/sample.py --dataset_path output/minhash/BELLE_dedup/lm_train.json --output_path dataset/en_mergy/lm_train_ch.json --ratio 0.434
python data_preprocess/merge.py --dataset_path dataset/en_mergy/lm_train.json --merge_from_path dataset/en_mergy/lm_train_ch.json --output_path dataset/en_mergy/lm_train.json

python data_preprocess/shuffle.py --dataset_path dataset/en_mergy/lm_train.json --output_path dataset/en_mergy/lm_train.json

# 加结束符
python data_preprocess/add_end_mark.py --dataset_path dataset/en_mergy/lm_train.json --output_path dataset/en_mergy/lm_train.json

python data_preprocess/count.py --dataset_path dataset/en_mergy/lm_train.json