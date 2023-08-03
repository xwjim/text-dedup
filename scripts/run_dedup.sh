#!/bin/bash

python -m text_dedup.minhash \
  --path "dataset/lawer/law_instruction.arrow" \
  --name "gl" \
  --split "train" \
  --cache_dir "./caches" \
  --output "output/minhash/lawer_dedup" \
  --column "input" \
  --local \
  --batch_size 10000