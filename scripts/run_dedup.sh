#!/bin/bash

python -m text_dedup.minhash \
  --path "Muennighoff/natural-instructions" \
  --name "gl" \
  --split "train" \
  --cache_dir "./cache" \
  --output "output/minhash/natural-instructions_dedup" \
  --column "inputs" \
  --batch_size 10000