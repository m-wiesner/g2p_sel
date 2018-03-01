#!/bin/bash

words="words.txt"
. ./utils/parse_options.sh

dir=$1

[ -f ${dir}/symb_er.txt ] && rm ${dir}/symb_er.txt
scores=( `find $dir -name "score.txt" | grep "${words}"` )

for f in ${scores[@]}; do
  grep "symbol errors:" $f | tail -1 | awk '{print $4}' |\
    sed 's/[()%]//g' | awk -v var=$f '{print var": "$0}' 
done >> ${dir}/symb_er.txt

[ -f ${dir}/str_er.txt ] && rm ${dir}/str_er.txt
for f in ${scores[@]}; do
  grep "string errors:" $f | tail -1 | awk '{print $4}' |\
    sed 's/[()%]//g' | awk -v var=$f '{print var": "$0}' 
done >> ${dir}/str_er.txt


[ -f ${dir}/F1.txt ] && rm ${dir}/F1.txt
for f in ${scores[@]}; do
  grep "F1" $f | tail -1 | awk '{print $3}' | sed 's/%//g' |\
    awk -v var=$f '{print var": "$0}'
done >> ${dir}/F1.txt 

[ -f ${dir}/Prec.txt ] && rm ${dir}/Prec.txt
for f in ${scores[@]}; do
  grep "PRECISION" $f | tail -1 | awk '{print $2}' | sed 's/%//g' |\
    awk -v var=$f '{print var": "$0}'
done >> ${dir}/Prec.txt 

[ -f ${dir}/Recall.txt ] && rm ${dir}/Recall.txt
for f in ${scores[@]}; do
  grep "RECALL" $f | tail -1 | awk '{print $3}' | sed 's/%//g' |\
    awk -v var=$f '{print var": "$0}'
done >> ${dir}/Recall.txt 

grep "${words}" ${dir}/symb_er.txt | awk -v var=${words} '{vals[NR]=$2; sum+=$2} END{std=0; for (i in vals){std+=((vals[i]-sum/NR)^2)/(NR)}; print var "Avg: " sum/NR; print var "Stdev: " sqrt(std) }' >> ${dir}/symb_er.txt
grep "${words}" ${dir}/str_er.txt | awk -v var=${words} '{vals[NR]=$2; sum+=$2} END{std=0; for (i in vals){std+=((vals[i]-sum/NR)^2)/(NR)}; print var "Avg: " sum/NR; print var "Stdev: " sqrt(std) }' >> ${dir}/str_er.txt
grep "${words}" ${dir}/F1.txt | awk -v var=${words} '{vals[NR]=$2; sum+=$2} END{std=0; for (i in vals){std+=((vals[i]-sum/NR)^2)/(NR)}; print var "Avg: " sum/NR; print var "Stdev: " sqrt(std) }' >> ${dir}/F1.txt
grep "${words}" ${dir}/Prec.txt | awk -v var=${words} '{vals[NR]=$2; sum+=$2} END{std=0; for (i in vals){std+=((vals[i]-sum/NR)^2)/(NR)}; print var "Avg: " sum/NR; print var "Stdev: " sqrt(std) }' >> ${dir}/Prec.txt
grep "${words}" ${dir}/Recall.txt | awk -v var=${words} '{vals[NR]=$2; sum+=$2} END{std=0; for (i in vals){std+=((vals[i]-sum/NR)^2)/(NR)}; print var "Avg: " sum/NR; print var "Stdev: " sqrt(std) }' >> ${dir}/Recall.txt

exit 0;
