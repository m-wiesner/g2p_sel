from __future__ import print_function
import numpy as np
import codecs
import argparse
import sys
import os
import json
from G2PSelection import BatchActiveSubset, SeqBatchActiveSubset, RandomSubset
from SubmodularObjective import FeatureObjective, CrossEntropyObjective, \
                                DecayingCrossEntropyObjective, \
                                FeatureCoverageObjective, TanhFeatureCoverageObjective, \
                                DSFObjective

# Replace with (from SubmodularObjective)

def parse_input():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", help="output file with selected g2p words")
    parser.add_argument("wordlist", help="path with a list of words")
    parser.add_argument("budget", help="The total budget allocated for words",
                        type=float)
    parser.add_argument("--max-size", help="The maximum number of test words "
                        "to process", type=int, default=1000000, action="store")
    parser.add_argument("--n-order", help="The maximum order n-gram to use in "
                        "calculating features", type=int, default=2,
                        action="store")
    parser.add_argument("--constraint", help="The type of constraint used "
                        "to limit the number of words selected.",
                        choices=['card', 'len', 'rootlen', 'freq', 'loglen'],
                        default='card', action="store")
    parser.add_argument("--rootlen", help="Inverse of power of length", type=int,
                        default=5, action="store")
    parser.add_argument("--root", help="Inverse power of root", type=float,
                        default=2, action="store")
    parser.add_argument("--subset-method", help="The method used to subset the "
                        "data.", choices=['BatchActive', 'Random', 'SeqBatchActive'],
                        default='BatchActive', action="store")
    parser.add_argument("--vectorizer", help="The type of vectorizer for "
                        "feature computation", choices=['tfidf', 'count'],
                        default='tfidf', action="store")
    parser.add_argument("--objective", help="The type of objective for "
                        " submodular optimization",
                        choices=['Feature', 'CrossEntropy',
                                 'DecayingCrossEntropy', 'FeatureCoverage',
                                 'TanhFeatureCoverage', 'DSF'],
                        default='Feature', action="store")
    parser.add_argument("--cost-select", help="Use cost in selection criteria",
                        action="store_true") 
    parser.add_argument("--append-ngrams", help="Append n,n-1,...,1-gram "
                        "features", action="store_true")
    parser.add_argument("--binarize-counts", help="Counts are 0, or 1.0",
                        action="store_true")


    return parser.parse_args()
    

def main():
    # Parse Aguments
    args = parse_input()
   
    # -------------------------------------
    # Nested cost functions
    # -------------------------------------
    def rootlen(w):
        return len(w) ** (1.0 / args.rootlen)
    
    def loglen(w):
        return np.log(len(w)+1.0)
   
    def card(w):
        return 1.0
    
    # -------------------------------------
    # Extract train and test word sets
    # -------------------------------------
    words = []
    freqs = {}
    # I might be passing in a wordlist file with counts
    if args.constraint == "freq":
        total_count = 0.0
        try:
            with codecs.open(args.wordlist, "r", encoding="utf-8") as f:
                for l in f:
                    if l.strip():
                        word, count = l.strip().split('\t', 1)
                        words.append(word)
                        freqs[word] = float(count)
                        total_count += float(count)
        except IndexError:
            print("Poorly formatted wordlist file. When using the "
                  "--constraint-type freq option words and their counts "
                  "from some corpus listed in two tab separated columns.",
                  file=sys.stderr)
       
        for w, c in freqs.iteritems():
            freqs[w] = -np.log2(c / total_count + sys.float_info.epsilon)        
         
    else:
        with codecs.open(args.wordlist, "r", encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    words.append(l.strip())
    

    def freq(w):
        try:
            return freqs[w]
        except KeyError:
            return 1.0
           
            
    def log_1plusx(x):
        return np.log(1 + x)

    def root(x):
        return np.power(x, (1./args.root))
    
    if len(words) > args.max_size:
        words = [words[i] for i in np.random.randint(len(words), size=(args.max_size,))]
        
    
    cost_functions = {'card': card, 'rootlen': rootlen, 'len': len,
                      'freq': freq, 'loglen': loglen}
    methods = {'BatchActive': BatchActiveSubset, 'Random': RandomSubset,
               'SeqBatchActive': SeqBatchActiveSubset}
    # -------------------------------------
    # Create the g2p selection "learners"
    # -------------------------------------
    objectives = {
                    'Feature': FeatureObjective,
                    'CrossEntropy': CrossEntropyObjective, 
                    'DecayingCrossEntropy': DecayingCrossEntropyObjective,
                    'FeatureCoverage': FeatureCoverageObjective,
                    'TanhFeatureCoverage': TanhFeatureCoverageObjective,
                    'DSF': DSFObjective
                 }

    fobj = objectives[args.objective](words, n_order=args.n_order,
                                      g=root, vectorizer=args.vectorizer,
                                      append_ngrams=args.append_ngrams,
                                      binarize_counts=args.binarize_counts) 
    
    print("----------------------------------")
    print("Budget: ", args.budget)
    print("Constraint: ", args.constraint)
    print("Objective: ", args.objective)
    print("Vectorizer: ", args.vectorizer)
    print("Words: ", args.wordlist)
    print("n-order: ", args.n_order)
    print("Subset Method: ", args.subset_method)
    print("Append: ", args.append_ngrams)
    print("Cost Select: ", args.cost_select)
    print("Binarize Counts: ", args.binarize_counts)
    print("----------------------------------")

    bas = methods[args.subset_method](
        fobj,
        args.budget,
        words, 
        cost=cost_functions[args.constraint],
        cost_select=args.cost_select
    )
    
    print("Begin Selection ...")
    selected_words = bas.run_lazy()
    if os.path.dirname(args.output) not in ('', '.'):
        if (not os.path.exists(os.path.dirname(args.output))):
            os.makedirs(os.path.dirname(args.output))
    
    
    # Write output log files and selected word list
    with codecs.open(args.output + ".vocab.log", "w", encoding="utf-8") as f:
        json.dump(fobj._ngram_vectorizer.vocabulary_, f,
                  separators=(',',': '),
                  indent=4, sort_keys=True)
    
    with codecs.open(args.output + ".probs.log", "w", encoding="utf-8") as f:
        json.dump(fobj.p.tolist(), f,
                 separators=(',', ': '),
                 indent=4, sort_keys=True)

    with codecs.open(args.output, "w", encoding="utf-8") as f:
        for w in selected_words:
            print(w, file=f)
    
if __name__ == "__main__":
    main()
