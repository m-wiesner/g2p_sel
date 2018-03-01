#!/bin/bash

odir=$1
words=$2
iters=$3

mkdir -p $odir

for ITER in `seq 1 $iters`; do
  awk '{print length(), $0}' $words | shuf | sort -n -r -s -k1,1 |\
    cut -d' ' -f2- > ${odir}/words.${ITER}.txt
done
