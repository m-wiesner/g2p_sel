#!/bin/bash

. ./path.sh
constraint="card"

. ./utils/parse_options.sh
if [ $# -eq 0 ]; then
  echo "Usage: ./local/word_subset2lexicon.sh <words> <lexicon> <num>"
  echo "    Ex.  ./local/word_subset2lexicon.sh words lexicon 30 > words_lexicon"
  exit 1;
fi

words=$1
lexicon=$2
num=$3

if [ $constraint == "card" ]; then
  head -n $num $words | awk '(NR==FNR) {
    counts[$1]+=1;
    a[$1][counts[$1]]=$0;
    next
    } 
    ($1 in a) {
      for (i in a[$1]) {
        print a[$1][i]
      }
    }' $lexicon -
fi

if [ $constraint == "len" ]; then
  awk -v var=$num '(NR==FNR) {
    counts[$1]+=1
    a[$1][counts[$1]]=$0;
    next
    } 
    ($1 in a) { 
      for (i in a[$1]) {
        print a[$1][i]
      }
      total_length+=length($1);
      if (total_length > var){
        exit
      }
    }' $lexicon $words
fi


