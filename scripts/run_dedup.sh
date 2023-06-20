#!/bin/bash

python -m text_dedup.minhash \
  --path "mrm8488/unnatural-instructions-full" \
  --name "gl" \
  --split "train" \
  --cache_dir "./cache" \
  --output "output/minhash/unnatural-instructions_dedup" \
  --column "instruction" \
  --batch_size 10000