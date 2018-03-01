#!/bin/bash

random=false
len=false
words="words.txt"
. ./utils/parse_options.sh

dir=$1

words=${words%.*} # Remove extension
budgets=( `find $dir -name "budget_*"` )
for b in ${budgets[@]}; do
  echo ${b}
  ./local/g2p_selection/average_ser.sh --words $words $b
done

if $random; then
  cat <(echo "budget str_er str_er_stdev F1 F1_stdev Prec Prec_stdev Recall Recall_stdev symb_er symb_er_stdev") \
      <(paste -d' ' <(for b in ${budgets[@]}; do echo ${b##*_}; done | sort -n) \
              <(grep "${words}Avg:" ${dir}/budget_*/str_er.txt | sed 's/budget_/budget=/' | sort -t'=' -n -k2 | awk '{print $2}') \
              <(grep "${words}Stdev:" ${dir}/budget_*/str_er.txt | sed 's/budget_/budget=/' | sort -t'=' -n -k2 | awk '{print $2}') \
              <(grep "${words}Avg:" ${dir}/budget_*/F1.txt | sed 's/budget_/budget=/' | sort -t'=' -n -k2 | awk '{print $2}') \
              <(grep "${words}Stdev:" ${dir}/budget_*/F1.txt | sed 's/budget_/budget=/' | sort -t'=' -n -k2 | awk '{print $2}') \
              <(grep "${words}Avg:" ${dir}/budget_*/Prec.txt | sed 's/budget_/budget=/' | sort -t'=' -n -k2 | awk '{print $2}') \
              <(grep "${words}Stdev:" ${dir}/budget_*/Prec.txt | sed 's/budget_/budget=/' | sort -t'=' -n -k2 | awk '{print $2}') \
              <(grep "${words}Avg:" ${dir}/budget_*/Recall.txt | sed 's/budget_/budget=/' | sort -t'=' -n -k2 | awk '{print $2}') \
              <(grep "${words}Stdev:" ${dir}/budget_*/Recall.txt | sed 's/budget_/budget=/' | sort -t'=' -n -k2 | awk '{print $2}') \
              <(grep "${words}Avg:" ${dir}/budget_*/symb_er.txt | sed 's/budget_/budget=/' | sort -t'=' -n -k2 | awk '{print $2}') \
              <(grep "${words}Stdev:" ${dir}/budget_*/symb_er.txt | sed 's/budget_/budget=/' | sort -t'=' -n -k2 | awk '{print $2}')) \
              > ${dir}/stats.${words}.txt
  
else
  cat <(echo "budget str_er F1 Prec Recall symb_er") \
      <(paste -d' ' <(for b in ${budgets[@]}; do echo ${b##*_}; done | sort -n) \
              <(grep "${words}Avg:" ${dir}/budget_*/str_er.txt | sed 's/budget_/budget=/' | sort -t'=' -n -k2 | awk '{print $2}') \
              <(grep "${words}Avg:" ${dir}/budget_*/F1.txt | sed 's/budget_/budget=/' | sort -t'=' -n -k2 | awk '{print $2}') \
              <(grep "${words}Avg:" ${dir}/budget_*/Prec.txt | sed 's/budget_/budget=/' | sort -t'=' -n -k2 | awk '{print $2}') \
              <(grep "${words}Avg:" ${dir}/budget_*/Recall.txt | sed 's/budget_/budget=/' | sort -t'=' -n -k2 | awk '{print $2}') \
              <(grep "${words}Avg:" ${dir}/budget_*/symb_er.txt | sed 's/budget_/budget=/' | sort -t'=' -n -k2 | awk '{print $2}')) \
              > ${dir}/stats.${words}.txt
fi

intervals=`for b in ${budgets[@]}; do echo ${b##*_}; done | sort -n`
ints=( $intervals )
max_budget=${ints[-1]}

if ! $len; then
  [ -f ${dir}/stdev ] && rm ${dir}/stdev
  touch ${dir}/stdev
  for f in ${dir}/budget_${max_budget}/trial.*/words.txt; do
    ./local/g2p_selection/compute_lengths.sh --subsetsizes "$intervals" $f |\
      paste -d' ' - ${dir}/stdev > ${dir}/stdev.tmp
    mv ${dir}/stdev.tmp ${dir}/stdev
  done

  awk '{
    for (i=1; i <= NF; i++) {
      vals[NR][i]=$i;
    }
  }
  END {
    for (i in vals) {
      sum=0;
      sum2=0;
      for (j in vals[i]) {
        sum+=vals[i][j];
        sum2+=(vals[i][j])^2
      }
      printf("%0.2f", sum/NF);
      printf(" %0.2f\n", sqrt((sum2/NF) - (sum/NF)^2));
    }
  }' ${dir}/stdev | paste -d' ' ${dir}/stats.${words}.txt \
                                <(cat <(echo "budget_letters budget_letters_stdev") -) \
                                > ${dir}/stats.tmp
  mv ${dir}/stats.tmp ${dir}/stats.${words}.txt

else
  sed -i 's/budget/budget_letters/g' ${dir}/stats.${words}.txt
  paste -d' ' ${dir}/stats.${words}.txt <(cat <(echo "budget") \
  <( for f in ${dir}/budget_*/trial.1/lexicon.orig; do awk '{print $1}' $f | LC_ALL=C sort -u | wc -l; done | sort -n)) > ${dir}/stats.tmp

  mv ${dir}/stats.tmp ${dir}/stats.${words}.txt
fi

