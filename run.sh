#!/bin/bash

###############################################################################
# OUTLINE:
#   Given a lexicon consisting of original words in the language and
# pronunications we do the following. First, we need a method of transforming 
# the original words, which may include some non-standard characters 
# (especially punctuation), into words suitable for G2P training.
#
# We use this method once on all of the words in the original lexicon.
#  
# Next, we select from the these words a subset according any of the methods 
# supported in 
#   
#   local/g2p_selection/g2p/select_g2p.py
# 
# 
# We also select, from the original lexicon the subset of the lexicon
# corresponding to these words.
#  
# Using the same transforming method we create a transformed lexicon subset, 
# which we then use for G2P training.
#
# We use the trained G2P on the transformed set 
#   T(original words \ selected words) to recover pronunciations for all
# words without pronunciations. We supplement this lexicon with the selected
# lexicon subset. 
###############################################################################

. ./path.sh

lexicon_file= 
romanized=false
stage=0
dictionary_dir=data/dict_201_haitian
lang=201_haitian
exp_affix=
intervals="40 100 200"
sel_intervals="8 16 24 32 40 64 80 100 128 200 256 400 500 512 750 1000 1024 1500 2000 2048 5000"
append_ngrams=true
objective="FeatureCoverage"
constraint="card"
n_order=4
max_budget=5000
diphthongs=/export/MStuDy/Matthew/LORELEI/kaldi/egs/universal_acoustic_model/s5_all_babel_llp/universal_phone_maps/diphthongs/201
tones=/export/MStuDy/Matthew/LORELEI/kaldi/egs/universal_acoustic_model/s5_all_babel_llp/universal_phone_maps/tones/201

. ./utils/parse_options.sh

data_dir=`dirname $dictionary_dir`
langdir=${data_dir}/lang_${lang}_${exp_affix}
###############################################################################
#                          Preprocess BABEL LEXICON
###############################################################################

extra_lex_opts=
if $romanized; then
  extra_lex_opts="--romanized"
fi

if [ $stage -le 0 ]; then
  ./local/prepare_lexicon.pl $extra_lex_opts --oov "<unk>" $lexicon_file $dictionary_dir
  
  # Save map from integer keys to original words and visa versa
  mkdir -p ${dictionary_dir}/g2p_sel
  
  # Remove <.*> words, <unk>, <silence>, etc.. since no grapheme-to-phoneme map
  # can be learned for these
  grep -v '<.*>' ${dictionary_dir}/lexicon.txt > ${dictionary_dir}/g2p_sel/filtered_lexicon.txt 
  
  cut -f2- ${dictionary_dir}/g2p_sel/filtered_lexicon.txt |\
    LC_ALL= sed 's/\t//g' |\
    paste <(cut -f1 ${dictionary_dir}/g2p_sel/filtered_lexicon.txt) - |\
    tee ${dictionary_dir}/g2p_sel/lexicon.txt | cut -f1 | LC_ALL=C sort -u \
    > ${dictionary_dir}/g2p_sel/words.txt
fi

###############################################################################
# Run G2P Selection 
###############################################################################

num_words=$(cat ${dictionary_dir}/g2p_sel/words.txt | wc -l)
selection_affix=${objective}-${constraint}-${n_order}
if $append_ngrams; then
  selection_affix=${selection_affix}+
fi

if [ $stage -le 1 ]; then
  # Replace $max_budget with $num_words to get full curve
  echo $sel_intervals
  ./local/run_g2p_selection.sh --constraint $constraint \
                               --n-order $n_order \
                               --method "BatchActive" \
                               --vectorizer "count" \
                               --objective $objective \
                               --score true \
                               --cost-select true \
                               --append-ngrams $append_ngrams \
                               --intervals "$sel_intervals" \
                               ${lang}/${selection_affix} \
                               ${dictionary_dir}/g2p_sel/words.txt \
                               ${dictionary_dir}/g2p_sel/lexicon.txt $max_budget
fi
exit
###############################################################################
# Given the lexicons produced, create the lang directories for different
# vocab sizes, training the LMs for the new vocabularies when necessary.
###############################################################################
if [ $stage -le 2 ]; then
  echo "------------------------------------------------------------"
  echo " Creating Dictionaries and Lang directories for G2P lexicons"
  echo "------------------------------------------------------------"
  for b in $intervals; do
    dict_dir=${dictionary_dir}_${exp_affix}_${b}
    mkdir -p ${dict_dir}
    
    # Check that the lexicon exists
    lex_path=${lang}/${exp_affix}/budget_${b}/trial.1/g2p/words/lexicon.txt
    
    # Haitian (trials 22, 38, 15 used chosen based on symb_er)
    #lex_path=${lang}/${exp_affix}/budget_${b}/trial.15/g2p/words/lexicon.txt
    [ ! -f ${lex_path} ] && echo "Expected $lex_path to exist" && exit 1 
   
    echo -e "<silence> SIL\n<unk> <oov>\n<noise> <sss>\n<v-noise> <vns>" \
      > ${dict_dir}/silence_lexicon.txt
  
    # Create the (universal) lexicon keeping only those words with non-empty prons
    ./local/prepare_universal_lexicon.py ${dict_dir}/nonsilence_lexicon.txt \
      <(awk '(NF > 1)' ${lex_path}) $diphthongs $tones
  
    # Add back in hesitation markers
    grep "<hes>" ${dictionary_dir}/lexicon.txt |\
     cat - ${dict_dir}/{,non}silence_lexicon.txt | LC_ALL=C sort \
     > ${dict_dir}/lexicon.txt
   
    # Create the rest of the dictionary    
    ./local/prepare_unicode_lexicon.py \
      --silence-lexicon ${dict_dir}/silence_lexicon.txt \
      ${dict_dir}/lexicon.txt ${dict_dir}
     
    ./utils/prepare_lang.sh --share-silence-phones true \
      ${dict_dir} "<unk>" ${dict_dir}/tmp.lang \
      ${langdir}_${b}
 
    ./local/phoneset_diff.sh ${langdir}_${b}/phones.txt \
      ../s5_all_babel_llp/data/lang_universalp/tri5_ali/phones.txt \
      > ${langdir}_${b}/missing_phones_map  
  
  done
fi

#echo "Stopped on line 132 to allow user to replace missing phones in " && \
#echo "${dictionary_dir}_{${intervals}}/missing_phones_map" && exit 1;

###############################################################################
# At this point, missing phones need to be manually mapped to existing phones.
# After this is done, comment out the "exit" in above line.
###############################################################################
for b in ${intervals}; do
  dict_dir=${dictionary_dir}_${exp_affix}_${b}
  if [ $stage -le 3 ]; then
    ./local/convert_dict.sh ${dict_dir}_universal \
      ${langdir}_${b}_universal \
      ${dict_dir} ../s5_all_babel_llp/data/dict_universal \
      ../s5_all_babel_llp/data/lang_universalp/tri5_ali \
      ${langdir}_${b}/missing_phones_map

    ###########################################################################
    # Train the LM
    ###########################################################################
    
    ./local/train_lms_srilm.sh --oov-symbol "<unk>" \
                               --train-text ${data_dir}/train/text \
                               --words-file ${langdir}_${b}_universal/words.txt \
                               ${data_dir}/ ${data_dir}/srilm_${exp_affix}_${b}

    ./local/arpa2G.sh ${data_dir}/srilm_${exp_affix}_${b}/lm.gz ${langdir}_${b}_universal ${langdir}_${b}_universal
  fi
  
  if [ $stage -le 4 ]; then
    ./utils/mkgraph.sh --self-loop-scale 1.0 ${langdir}_${b}_universal \
      ../s5_all_babel_llp/exp/chain_cleaned/tdnn_sp_bi \
      ../s5_all_babel_llp/exp/chain_cleaned/tdnn_sp_bi/graph_${lang}_g2p_${exp_affix}_${b}
  fi
done

