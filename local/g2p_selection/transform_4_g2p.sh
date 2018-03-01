#!/bin/bash

lex=$1

awk -F'\t' '{for (i=2; i<NF; i++){printf("%s\t", $i)} if(NF >= 2){printf("%s\n", $NF)}}' $lex |\
  LC_ALL= sed 's/\t//g' |\
  awk -F'\t' '(NR==FNR) {
    a[NR]=$1;
    entries+=1;
    next
    } (NR>FNR) {
      a[FNR]=a[FNR]"\t"$0
    } END {
      for(i=1; i<=entries; i++) {
        printf("%s\n", a[i])
      }
    }' <(cut -f1 $lex | LC_ALL= sed 's/./\L&/g;s/_/ /g;s/\///g') -
