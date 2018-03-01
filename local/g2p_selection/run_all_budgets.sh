#!/bin/bash
. ./path.sh

###############################################################################
# Outline:
#   1. Take in list of actual words for which we want pronunciations
#   2. Remove from this list all words used in training of G2P since 
#      we actually do have those pronunciations
#   3. Transform the remaining list of words according to 
#      local/g2p_selection/transform_4_g2p.sh
#   4. Obtain pronunciations for all of these words
#   5. Give the original, untransformed words the learned pronunciations of the
#      transformed words
#   6. Append to this resulting lexicon the untransformed training lexicon
#      (as an option)
###############################################################################
cmd="run.pl"
nbest=1
score=true

# Turkish
ref_lex=105_turkish/LM_resources/lexicon.txt
words=105_turkish/LM_resources/words.txt

. ./utils/parse_options.sh
if [ $# -eq 0 ]; then
  echo "Usage: ./local/run_all_budgets.sh [--opts] <dir>"
  echo "    --nbest <int> : # of nbest pronunciation hypothesis. Default = 1" 
  echo "    --cmd \"queue.pl\" : submit jobs to grid. Default = \"run.pl\""
  echo "    --score true,false : option to score created lexicons"
  exit 1
fi

dir=$1
trials=( `find $dir -name "trial.*"` )
num_trials=${#trials[@]}

cwd=`readlink -f .`
mkdir -p ${dir}/g2p/log

for i in `seq 1 $num_trials`; do
  trial=`readlink -f ${trials[$(($i-1))]}`
  echo $trial 
  cd ${dir}/g2p
  if [ ! -L trial.$i ] || [ ! -e trial.$i ]; then
    if [ -L trial.$i ]; then
      rm trial.$i
    fi
    ln -s $trial trial.$i
  fi 
  cd $cwd
done  

$cmd ITER=1:${num_trials} ${dir}/g2p/log/run.ITER.log \
  ./local/g2p_selection/run_single_g2p.sh --nbest $nbest --score $score ${dir}/g2p/trial.ITER $words $ref_lex || exit 1

exit 0
