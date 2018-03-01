from __future__ import print_function
import sys
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy import stats
import scipy.sparse as sparse
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pdb


class FeatureObjective(object):
    def __init__(self, wordlist, n_order=1, g=np.sqrt, vectorizer='tfidf',
                 append_ngrams=True, binarize_counts=False):
        self.wordlist = wordlist
        self._ngram_vectorizer = None
        self._g = g
        self.n_order = n_order
        self.append_ngrams = append_ngrams
        self.binarize_counts = binarize_counts
        
        vectorize_methods = {'tfidf': TfidfVectorizer, 'count': CountVectorizer}
        self.vectorizer = vectorize_methods[vectorizer]
        
        self.word_features = None
        self.subset = None
        self.p = None
        
        self.get_tfidf_vectorizers()
        self.get_word_features()
    
    
    def get_tfidf_vectorizers(self):
        '''
            Input: word_tokens in the format output by words_to_tokens
            Output: A sparse matrix of tfidf features where each row a
                    "document" in the corpus.
        '''
        min_ngram = self.n_order
        if self.append_ngrams:
            min_ngram = 1
        ngram_range = (min_ngram, self.n_order)

        vectorizer = self.vectorizer(analyzer="char_wb",
                                    encoding="utf-8",
                                    strip_accents=None,
                                    ngram_range=ngram_range)
        
        vectorizer.fit(self.wordlist)
        self._ngram_vectorizer = vectorizer


    def set_subset(self, idxs):
        self.subset = self.word_features[idxs,:].sum(axis=0)
   
    
    def add_to_subset(self, idx):
        self.subset += self.word_features[idx, :]

        
    def run(self, idx_new):
        return self._g(self.word_features[idx_new, :] + self.subset).sum()

    
    def get_word_features(self):
        self.word_features = self._ngram_vectorizer.transform(self.wordlist)
        if self.binarize_counts:
            self.word_features[self.word_features > 0 ] = 1.0
        self.p = self.word_features.sum(axis=0) / float(self.word_features.sum())


    def reset(self):
        self.n_order += 1
        self._ngram_vectorizer = None

        self.word_features = None
        self.subset = None

        self.get_tfidf_vectorizers()
        self.get_word_features()

    def compute_entropy(self):
        prob_vec = self.subset / float(self.subset.sum())
        return -(np.multiply(prob_vec, np.log2(prob_vec + sys.float_info.epsilon))).sum()


    def compute_kl(self):
        prob_vec = (1.0 + self.subset) / (float(self.subset.sum()) + self.subset.shape[1])
        return np.multiply(self.p, np.log2(self.p + sys.float_info.epsilon) - np.log2(prob_vec)).sum() 


class DecayingCrossEntropyObjective(FeatureObjective):
    def run(self, idx_new):
        if not idx_new:
            return np.multiply(self.p, np.log2(self.subset + 1)).sum()
        
        return np.multiply(self.p, np.log2(self.word_features[idx_new, :] + self.subset + 1)).sum()


    def smooth_p(self, factor):
        self.p = (1 - factor) * self.p + factor * ( 1. / self.p.shape[1]) 
    
    
class CrossEntropyObjective(FeatureObjective):
    def __init__(self, wordlist, n_order=1, g=np.sqrt, vectorizer='tfidf',
                 append_ngrams=True, prune=False, prune_thresh=0.0000,
                 binarize_counts=False):
        self.wordlist = wordlist
        self._ngram_vectorizer = None
        self._g = g
        self.n_order = n_order
        self.append_ngrams = append_ngrams
        self.prune = prune
        self.prune_thresh = prune_thresh

        vectorize_methods = {'tfidf': TfidfVectorizer, 'count': CountVectorizer}
        self.vectorizer = vectorize_methods[vectorizer]
        
        self.word_features = None
        self.subset = None
        self.p = None
    
        self.get_tfidf_vectorizers()
        self.get_word_features()
   
        
    def get_tfidf_vectorizers(self):
        '''
            Input: word_tokens in the format output by words_to_tokens
            Output: A sparse matrix of tfidf features where each row a
                    "document" in the corpus.
        '''
        min_ngram = self.n_order
        if self.append_ngrams:
            min_ngram = 1
        ngram_range = (min_ngram, self.n_order)

        prune_thresh = 0.0
        if self.prune:
            prune_thresh = self.prune_thresh

        vectorizer = self.vectorizer(analyzer="char_wb",
                                    encoding="utf-8",
                                    strip_accents=None,
                                    ngram_range=ngram_range,
                                    min_df=prune_thresh)
        
        vectorizer.fit(self.wordlist)
        self._ngram_vectorizer = vectorizer

        
class FeatureCoverageObjective(FeatureObjective):
    def __init__(self, *args, **kwargs):
        self.total_counts = None
        self.K = None
        kwargs["binarize_counts"] = True
        super(FeatureCoverageObjective, self).__init__(*args, **kwargs)
          

    def get_word_features(self):
        self.word_features = self._ngram_vectorizer.transform(self.wordlist)
        self.word_features[self.word_features > 0 ] = 1.0
        self.total_counts = self.word_features.sum(axis=0)
        self.p = self.total_counts / float(self.word_features.sum())
        self.K = self.total_counts.sum()


    def run(self, idx_new):
        vec = self.subset + self.word_features[idx_new, :]
        p_vec = np.squeeze(np.asarray(self.total_counts))
        vec = np.squeeze(np.asarray(vec))
        return self.K - np.multiply(p_vec, (1.0 / (8.0 ** vec))).sum()

    
class TanhFeatureCoverageObjective(FeatureObjective):
    def __init__(self, *args, **kwargs):
        self.total_counts = None
        self.alphas = None
        self.Beta = kwargs.get("Beta", 0.99999)
        kwargs.pop("Beta", None)
        super(TanhFeatureCoverageObjective, self).__init__(*args, **kwargs)
          

    def get_word_features(self):
        self.word_features = self._ngram_vectorizer.transform(self.wordlist)
        self.word_features[self.word_features > 0 ] = 1.0
        self.total_counts = self.word_features.sum(axis=0)
        self.p = self.total_counts / float(self.word_features.sum())
        self.alphas = (1.0 / (2 * self.total_counts)) * np.log((1 + self.Beta)/(1 - self.Beta))


    def run(self, idx_new):
        if not idx_new:
            vec = np.squeeze(np.asarray(self.subset))
            return np.multiply(np.squeeze(np.asarray(self.total_counts)),
                        np.tanh(np.multiply(np.squeeze(np.asarray(self.alphas)), vec))).sum()

        vec = self.subset + self.word_features[idx_new, :]
        p_vec = np.squeeze(np.asarray(self.total_counts))
        vec = np.squeeze(np.asarray(vec))
        return np.multiply(p_vec, np.tanh(np.multiply(np.squeeze(np.asarray(self.alphas)), vec))).sum()


class DSFObjective(FeatureObjective):
    def __init__(self, *args, **kwargs):
        self._order_idxs = {}
        self.W = None
        self.total_counts = None
        kwargs["append_ngrams"] = True
        kwargs["n_order"] = 3
        super(DSFObjective, self).__init__(*args, **kwargs)
          

    def get_word_features(self):
        self.word_features = self._ngram_vectorizer.transform(self.wordlist)
       
        if self.binarize_counts:
            self.word_features[self.word_features > 0 ] = 1.0
        self.total_counts = self.word_features.sum(axis=0)
        self.p = self.total_counts / float(self.word_features.sum())

        for o in range(self.n_order, 0, -1):
            self._order_idxs[o] = []
            for k, v in self._ngram_vectorizer.vocabulary_.iteritems():
                if len(k) == o:
                    self._order_idxs[o].append(v)
            self._order_idxs[o] = np.asarray(self._order_idxs[o])
       
        idx_to_ngram = self._ngram_vectorizer.get_feature_names()
        row_idx = []
        col_idx = []
        data = []
        
        # Sort by ngram order
        three_grams = [idx_to_ngram[i] for i in self._order_idxs[3]]
        for tg in three_grams:
            one_grams = [i for i in self._order_idxs[1] if idx_to_ngram[i] == tg[1]]
            row_idx.extend(one_grams)
            col_idx.extend(len(one_grams) * [self._ngram_vectorizer.vocabulary_[tg]])
            # The 3-gram weight should be a function of the 4-gram freq e.g.
            # This way the more common occurrence is weighted more. 
            data.extend(len(one_grams) * [self.total_counts[0, self._ngram_vectorizer.vocabulary_[tg]]])
            #data.extend(len(o_min_1_grams) * [1.0]) 
        
        self.W = sparse.csr_matrix((data, (row_idx, col_idx)),
            shape=(self.word_features.shape[1], self.word_features.shape[1])) 
        
        
    def add_to_subset(self, idx):
        # Only add max order ngram features in current word
        n = min([self.n_order, len(self.wordlist[idx]) + 2])
        self.subset[:, self._order_idxs[n]] += self.word_features[idx, self._order_idxs[n]]


    def run(self, idx_new):
        vec = self.subset.copy()
        n = min([self.n_order, len(self.wordlist[idx_new]) + 2])
        vec[:, self._order_idxs[n]] += self.word_features[idx_new, self._order_idxs[n]]

        counts = np.zeros(vec.shape)
        counts[:, self._order_idxs[self.n_order]] = vec[:, self._order_idxs[self.n_order]]
        vec = self._g(vec)
        counts = self.W.dot(self._g(counts).T).T
        return np.multiply(np.squeeze(np.asarray(self.total_counts)), self._g(counts)).sum() 


#class DSFObjective(FeatureObjective):
#    def __init__(self, *args, **kwargs):
#        self._order_idxs = {}
#        self.W = None
#        self.total_counts = None
#        self.K = None
#        kwargs["append_ngrams"] = True
#        kwargs["g"] = self.coverage_fun
#        super(DSFObjective, self).__init__(*args, **kwargs)
#          
#
#    def get_word_features(self):
#        self.word_features = self._ngram_vectorizer.transform(self.wordlist)
#       
#        if self.binarize_counts:
#            self.word_features[self.word_features > 0 ] = 1.0
#        self.total_counts = self.word_features.sum(axis=0)
#        self.p = self.total_counts / float(self.word_features.sum())
#        self.K = self.total_counts.sum()
#
#        for o in range(self.n_order, 0, -1):
#            self._order_idxs[o] = []
#            for k, v in self._ngram_vectorizer.vocabulary_.iteritems():
#                if len(k) == o:
#                    self._order_idxs[o].append(v)
#            self._order_idxs[o] = np.asarray(self._order_idxs[o])
#       
#        idx_to_ngram = self._ngram_vectorizer.get_feature_names()
#        row_idx = []
#        col_idx = []
#        data = []
#        
#        # Sort by ngram order
#        for o in range(self.n_order, 1, -1):
#            ograms = [idx_to_ngram[i] for i in self._order_idxs[o]]
#            for og in ograms:
#                o_min_1_grams = [i for i in self._order_idxs[o-1] if idx_to_ngram[i] in og]
#                row_idx.extend(o_min_1_grams)
#                col_idx.extend(len(o_min_1_grams) * [self._ngram_vectorizer.vocabulary_[og]])
#                # The 3-gram weight should be a function of the 4-gram freq e.g.
#                # This way the more common occurrence is weighted more. 
#                data.extend(len(o_min_1_grams) * [self.total_counts[0, self._ngram_vectorizer.vocabulary_[og]]])
#                #data.extend(len(o_min_1_grams) * [1.0]) 
#        
#        self.W = sparse.csr_matrix((data, (row_idx, col_idx)),
#            shape=(self.word_features.shape[1], self.word_features.shape[1])) 
#        
#        
#    
#    def add_to_subset(self, idx):
#        # Only add max order ngram features in current word
#        n = min([self.n_order, len(self.wordlist[idx]) + 2])
#        self.subset[:, self._order_idxs[n]] += self.word_features[idx, self._order_idxs[n]]
#
#
#    def coverage_fun(self, x):
#        p_vec = np.squeeze(np.asarray(self.total_counts))
#        vec = np.squeeze(np.asarray(x))
#        return np.matrix(1.0 - (1.0 / (8.0 ** vec)))
#
#
#    def run(self, idx_new):
#        vec = self.subset.copy()
#        n = min([self.n_order, len(self.wordlist[idx_new]) + 2])
#        vec[:, self._order_idxs[n]] += self.word_features[idx_new, self._order_idxs[n]]
#
#        counts = np.zeros(vec.shape)
#        counts[:, self._order_idxs[self.n_order]] = vec[:, self._order_idxs[self.n_order]]
#        for n in range(self.n_order, 1, -1):
#            vec = self._g(vec)
#            counts = self.W.dot(self._g(counts).T).T
#            if n >= 2:
#                counts[:, self._order_idxs[n-1]] += vec[:, self._order_idxs[n-1]]
#            
#        
#        return np.multiply(np.squeeze(np.asarray(self.total_counts)), self._g(counts)).sum() 



#class DSFObjective(FeatureObjective):
#    def __init__(self, *args, **kwargs):
#        self._order_idxs = {}
#        self.W = None
#        self.total_counts = None
#        kwargs["append_ngrams"] = True
#        super(DSFObjective, self).__init__(*args, **kwargs)
#          
#
#    def get_word_features(self):
#        self.word_features = self._ngram_vectorizer.transform(self.wordlist)
#       
#        if self.binarize_counts:
#            self.word_features[self.word_features > 0 ] = 1.0
#        self.total_counts = self.word_features.sum(axis=0)
#        self.p = self.total_counts / float(self.word_features.sum())
#
#        for o in range(self.n_order, 0, -1):
#            self._order_idxs[o] = []
#            for k, v in self._ngram_vectorizer.vocabulary_.iteritems():
#                if len(k) == o:
#                    self._order_idxs[o].append(v)
#            self._order_idxs[o] = np.asarray(self._order_idxs[o])
#       
#        idx_to_ngram = self._ngram_vectorizer.get_feature_names()
#        row_idx = []
#        col_idx = []
#        data = []
#        
#        # Sort by ngram order
#        for o in range(self.n_order, 1, -1):
#            ograms = [idx_to_ngram[i] for i in self._order_idxs[o]]
#            for og in ograms:
#                o_min_1_grams = [i for i in self._order_idxs[o-1] if idx_to_ngram[i] in og]
#                row_idx.extend(o_min_1_grams)
#                col_idx.extend(len(o_min_1_grams) * [self._ngram_vectorizer.vocabulary_[og]])
#                # The 3-gram weight should be a function of the 4-gram freq e.g.
#                # This way the more common occurrence is weighted more. 
#                data.extend(len(o_min_1_grams) * [self.total_counts[0, self._ngram_vectorizer.vocabulary_[og]]])
#                #data.extend(len(o_min_1_grams) * [1.0]) 
#        
#        self.W = sparse.csr_matrix((data, (row_idx, col_idx)),
#            shape=(self.word_features.shape[1], self.word_features.shape[1])) 
#        
#        
#    
#    def add_to_subset(self, idx):
#        # Only add max order ngram features in current word
#        n = min([self.n_order, len(self.wordlist[idx]) + 2])
#        self.subset[:, self._order_idxs[n]] += self.word_features[idx, self._order_idxs[n]]
#
#
#    def run(self, idx_new):
#        vec = self.subset.copy()
#        n = min([self.n_order, len(self.wordlist[idx_new]) + 2])
#        vec[:, self._order_idxs[n]] += self.word_features[idx_new, self._order_idxs[n]]
#
#        counts = np.zeros(vec.shape)
#        counts[:, self._order_idxs[self.n_order]] = vec[:, self._order_idxs[self.n_order]]
#        for n in range(self.n_order, 1, -1):
#            vec = self._g(vec)
#            counts = self.W.dot(self._g(counts).T).T
#            if n >= 2:
#                counts[:, self._order_idxs[n-1]] += vec[:, self._order_idxs[n-1]]
#            
#        
#        return np.multiply(np.squeeze(np.asarray(self.total_counts)), self._g(counts)).sum() 



#class DSFObjective(FeatureObjective):
#    def __init__(self, *args, **kwargs):
#        self._order_idxs = {}
#        self.W = None
#        kwargs["append_ngrams"] = True
#        super(DSFObjective, self).__init__(*args, **kwargs)
#          
#
#    def get_word_features(self):
#        self.word_features = self._ngram_vectorizer.transform(self.wordlist)
#       
#        for o in range(self.n_order, 0, -1):
#            self._order_idxs[o] = []
#            for k, v in self._ngram_vectorizer.vocabulary_.iteritems():
#                if len(k) == o:
#                    self._order_idxs[o].append(v)
#            self._order_idxs[o] = np.asarray(self._order_idxs[o])
#       
#        idx_to_ngram = self._ngram_vectorizer.get_feature_names()
#        row_idx = []
#        col_idx = []
#        data = []
#        
#        # Sort by ngram order
#        for o in range(self.n_order, 1, -1):
#            ograms = [idx_to_ngram[i] for i in self._order_idxs[o]]
#            for og in ograms:
#                o_min_1_grams = [i for i in self._order_idxs[o-1] if idx_to_ngram[i] in og]
#                row_idx.extend(o_min_1_grams)
#                col_idx.extend(len(o_min_1_grams) * [self._ngram_vectorizer.vocabulary_[og]])
#                data.extend(len(o_min_1_grams) * [1.0]) 
#        
#        self.W = sparse.csr_matrix((data, (row_idx, col_idx)),
#            shape=(self.word_features.shape[1], self.word_features.shape[1])) 
#        
#        self.word_features = self._ngram_vectorizer.transform(self.wordlist)
#        if self.binarize_counts:
#            self.word_features[self.word_features > 0 ] = 1.0
#        self.total_counts = self.word_features.sum(axis=0)
#        self.p = self.total_counts / float(self.word_features.sum())
#
#    
#    def add_to_subset(self, idx):
#        # Only add max order ngram features in current word
#        n = min([self.n_order, len(self.wordlist[idx]) + 2])
#        self.subset[:, self._order_idxs[n]] += self.word_features[idx, self._order_idxs[n]]
#
#
#    def run(self, idx_new):
#        vec = self.subset.copy()
#        n = min([self.n_order, len(self.wordlist[idx_new]) + 2])
#        vec[:, self._order_idxs[n]] += self.word_features[idx_new, self._order_idxs[n]]
#
#        counts = np.zeros(vec.shape)
#        counts[:, self._order_idxs[self.n_order]] = vec[:, self._order_idxs[self.n_order]]
#        for n in range(self.n_order, 1, -1):
#            vec = self._g(vec)
#            counts = self.W.dot(self._g(counts).T).T
#            if n >= 2:
#                counts[:, self._order_idxs[n-1]] += vec[:, self._order_idxs[n-1]]
#       
#        return self._g(counts).sum() 

# I have to reimplement this. The reference is in BatchActiveLearner_Graph.py
class FacilityLocationObjective(object):
    def __init__(self):
        pass
