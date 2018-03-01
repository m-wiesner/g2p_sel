#!/bin/bash

. ./path.sh

order=3
intervals="8 16 24 32 40 64 80 100 128 200 256 400 512 750 1024 2048"

. ./utils/parse_options.sh

words=$1
subset=$2


for subsize in ${intervals[@]}; do
  python local/g2p/compute_entropy.py $words \
                                    <(head -n $subsize $subset) $order
done
