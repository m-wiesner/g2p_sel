#!/bin/bash
. ./path.sh

words_orig=201_haitian/LM_resources.words_dev10h.pem.txt
words=201_haitian/LM_resources/words_dev10h.pem.lower.sorted.txt
g2p=201_haitian/Random_2/budget_40/lexicon.77.lexicon.g2p
g2p_sel=201_haitian/BatchActive_2_card_feat_order_3/budget_40/lexicon.1.lexicon.g2p
odir=201_haitian/LM_resources/dict_dev10h_40w
sort_keys=201_haitian/LM_resources/words_sort_keys
orig_keys=201_haitian/LM_resources/words_orig_keys

. ./utils/parse_options.sh

dir=`dirname $words_orig`

if [ ! -f $sort_keys ]; then
  awk '{print $1, NR}' $words_orig | LC_ALL= sed 's/./\L&/g' |\
    LC_ALL=C sort > ${dir}/words_sort_keys
fi

if [ ! -f $orig_keys ]; then
  awk '{print NR, $1}' $words_orig > ${dir}/words_orig_keys
fi

./local/apply_g2p.sh --nbest 1 $words $g2p ${odir}/g2p
cat ${odir}/g2p/lexicon_out.* | cut -f1,3 | LC_ALL=C sort > ${odir}/lexicon.g2p.txt
cat ${odir}/lexicon.g2p.txt | ./utils/apply_map.pl -f 1 $sort_keys |\
  ./utils/apply_map.pl -f 1 $orig_keys |\
   LC_ALL= sed 's/ /\t/' > ${odir}/lexicon.txt

./local/apply_g2p.sh --nbest 1 $words $g2p ${odir}/g2p_sel
cat ${odir}/g2p_sel/lexicon_out.* | cut -f1,3 | LC_ALL=C sort > ${odir}/lexicon.g2p_sel.txt
cat ${odir}/lexicon.g2p_sel.txt | ./utils/apply_map.pl -f 1 $sort_keys |\
  ./utils/apply_map.pl -f 1 $orig_keys |\
   LC_ALL= sed 's/ /\t/' > ${odir}/lexicon.sel.xt
