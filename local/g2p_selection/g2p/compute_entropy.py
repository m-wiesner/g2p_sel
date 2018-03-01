from __future__ import print_function
import numpy as np
import codecs
import argparse
import sys
import os
from G2PSelection import BatchActiveSubset, RandomSubset
from SubmodularObjective import FeatureObjective, EntropyObjective

if len(sys.argv[1:]) == 0:
    print("Usage: python ./local/g2p/compute_entropy.py <wordlist> <subset_list> <n-order>")
    sys.exit(1)

words = []
with codecs.open(sys.argv[1], "r", encoding="utf-8") as f:
    for l in f:
        if l.strip():
            words.append(l.strip())


fobj = EntropyObjective(words, n_order=int(sys.argv[3]), g=np.sqrt,
                        vectorizer='count')

fobj.get_word_features()
subset = []
with codecs.open(sys.argv[2], "r", encoding="utf-8") as f:
    for l in f:
        if l.strip():
            subset.append(words.index(l.strip()))

idx_last = subset.pop()
fobj.set_subset(subset)
print("(Budget, Entropy): ", len(subset) + 1, fobj.run(idx_last))



