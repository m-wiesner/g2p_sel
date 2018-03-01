#!/bin/bash

. ./path.sh

subsetsizes="8 16 24 32 40 64 80 100 128 200 256 400 512 750 1024 2048 4096 8192 16384 32218"

. ./utils/parse_options.sh

subset=$1

unset LC_ALL
for subsize in ${subsetsizes[@]}; do
  awk '{sum+=length()} END{print sum}' <(head -n $subsize $subset)
done
export LC_ALL=C
