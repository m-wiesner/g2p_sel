from __future__ import print_function
import numpy as np
from Queue import PriorityQueue, Empty
from collections import deque
import sys
import pdb


MAX_VALUE=999999999.0


class BatchActiveSubset(object):
    '''
        BatchActiveSubset:
            A class controlling the parameters of the batch active learner.
            It takes as input the submodular objective function, the budget,
            which is the constraint in the optimization problem, a list of words
            used to learn some feature extractors or edge similarity for
            instance, a list of words from which we will select upto budget's
            worth, and a few optional arguments that control the selection.
            
            Inputs: 
                objective -- The submodular objective to optimize. This is
                             implemented as a class
                budget    -- The contraint in the submodular optimization problem
                wordlist  -- A list of words from which to select budget's
                             worth
                cost=1 -- The cost of select each word for g2p training. The cost
                          could be things such as the length, some power of the
                          length, the word frequency, etc..
    '''
    def __init__(self, objective, budget, wordlist,
                cost=lambda x: 1, cost_select=False, window=9):
        self.objective = objective # An objective object
        self.budget = budget
        self.test_wordlist = wordlist
        self.cost = cost
        self.cost_select = cost_select 
        self.window = window
    
 
    def run_lazy(self):
        S = []
        set_value = 0.0
        total_cost = 0.0
        remaining_budget = self.budget
        kl = deque(maxlen=self.window)

        # Initialize Priority Queue
        upper_bounds = PriorityQueue()
        self.objective.set_subset(S)
        for i, w in enumerate(self.test_wordlist):
            if (self.cost(w) <= self.budget):
                upper_bounds.put((-MAX_VALUE, i))    
        
        # Main loop (Continue until budget is reached, or no elements remain) 
        while total_cost < self.budget and not upper_bounds.empty():
            alpha_idx, idx = upper_bounds.get(False)
           
            if (self.cost(self.test_wordlist[idx]) > remaining_budget):
                continue;
                 
            new_set_value = self.objective.run(idx)
            gain = (new_set_value - set_value)
            if gain < 0:
                pdb.set_trace()
            if self.cost_select:
                gain /= self.cost(self.test_wordlist[idx])
          
            # This is for the case when idx is the last element 
            try: 
                max_val, idx_max = upper_bounds.get(False)
                upper_bounds.put((max_val, idx_max))
            except Empty:
                max_val = 0.0 

            if (gain >= -max_val):
                S.append(idx)
                #self.objective.set_subset(S)
                self.objective.add_to_subset(idx)
                total_cost += self.cost(self.test_wordlist[idx])
                set_value = new_set_value

                print(self.test_wordlist[idx].encode('utf-8'), "Consumed: ", total_cost, "/", self.budget,
                    ":  +", self.cost(self.test_wordlist[idx]), end=" : ")
                print("Set value: ", set_value, end=" : ")
                print("Gain: ", gain, end=" : ")
 
                remaining_budget = self.budget - total_cost
                print("Remaining Candidates: ", upper_bounds.qsize())
                #kl.append(self.objective.compute_kl())
                #print("KL: ", np.mean(kl), np.std(kl))

            else:
                upper_bounds.put((-gain, idx))
                
        return [self.test_wordlist[idx] for idx in S]


class SeqBatchActiveSubset(object):
    def __init__(self, objective, budget, wordlist,
                cost=lambda x: 1, cost_select=False,
                entropy_thresh=0.03, entropy_window=9, max_n_order=3,
                smoothing_factor=0.2, smooth_iters=4, max_smooth_iters=10):
        self.objective = objective # An objective object
        self.budget = budget
        self.test_wordlist = wordlist
        self.cost = cost
        self.cost_select = cost_select 
        self.entropy_thresh = entropy_thresh
        self.max_n_order = max_n_order
        self.entropy_window = entropy_window
        self.smoothing_factor = smoothing_factor
        self.smooth_iters = smooth_iters
        self.max_smooth_iters = max_smooth_iters

    def run_lazy(self):
        S = []
        set_value = 0.0
        total_cost = 0.0
        smooth_iters = 0
        entropy = deque(maxlen=self.entropy_window)
        remaining_budget = self.budget
         
        # Initialize Priority Queue
        upper_bounds = PriorityQueue()
        self.objective.set_subset(S)
        for i, w in enumerate(self.test_wordlist):
            if (self.cost(w) <= self.budget):
                upper_bounds.put((-MAX_VALUE, i))    
        
        # Main loop
        while total_cost < self.budget and not upper_bounds.empty():
            alpha_idx, idx = upper_bounds.get(False)
           
            if (self.cost(self.test_wordlist[idx]) > remaining_budget):
                continue;
                 
            new_set_value = self.objective.run(idx)
            gain = (new_set_value - set_value)
            if self.cost_select:
                gain /= self.cost(self.test_wordlist[idx])
          
            # This is for the case when idx is the last element 
            try: 
                max_val, idx_max = upper_bounds.get(False)
                upper_bounds.put((max_val, idx_max))
            except Empty:
                max_val = 0.0 

            if (gain >= -max_val):
                S.append(idx)
                self.objective.set_subset(S)
                total_cost += self.cost(self.test_wordlist[idx])
                set_value = new_set_value

                print(self.test_wordlist[idx].encode('utf-8'), "Consumed: ", total_cost, "/", self.budget,
                    ":  +", self.cost(self.test_wordlist[idx]), end=" : ")
                print("Set value: ", set_value, end=" : ")
                print("Gain: ", gain, end=" : ")
 
                remaining_budget = self.budget - total_cost
                print("Remaining Candidates: ", upper_bounds.qsize(), end=" : ")
                
                # Compute Entropy, Check for convergence ...
                entropy.append(self.objective.compute_kl())
                print("KL: ", np.mean(entropy), np.std(entropy))
                try:
                    if np.std(entropy) < self.entropy_thresh and len(entropy) == self.entropy_window and smooth_iters <= self.smooth_iters:
                        self.objective.smooth_p(self.smoothing_factor)
                        smooth_iters += 1
                        print("*KL converged: ", np.std(entropy), "Smoothing distribution.")
                        set_value = self.objective.run(None) 
                        entropy.clear()
                    
                        # Initialize Priority Queue
                        upper_bounds_ = PriorityQueue()
                        while not upper_bounds.empty():
                            alpha_ub, idx_ub = upper_bounds.get(False)
                            new_set_value = self.objective.run(idx_ub)
                            gain_ub = (new_set_value - set_value)
                            if self.cost_select:
                                gain_ub /= self.cost(self.test_wordlist[idx])
                            upper_bounds_.put((-gain_ub, idx_ub))

                        upper_bounds = upper_bounds_
                        continue
                except AttributeError:
                    pass
                if np.std(entropy) < self.entropy_thresh and self.objective.n_order < self.max_n_order and len(entropy) == self.entropy_window:
                    print("*KL converged: ", np.std(entropy), "Increasing n-gram order to", self.objective.n_order + 1)
                    self.objective.reset()
                    self.objective.set_subset(S)
                    set_value = self.objective.run(None) 
                    entropy.clear()
                    smooth_iters = 0
                    
                    # Initialize Priority Queue
                    upper_bounds_ = PriorityQueue()
                    while not upper_bounds.empty():
                        alpha_ub, idx_ub = upper_bounds.get(False)
                        new_set_value = self.objective.run(idx_ub)
                        gain_ub = (new_set_value - set_value)
                        if self.cost_select:
                            gain_ub /= self.cost(self.test_wordlist[idx])
                        upper_bounds_.put((-gain_ub, idx_ub))

                    upper_bounds = upper_bounds_
                elif np.std(entropy) < self.entropy_thresh and len(entropy) == self.entropy_window and smooth_iters <= self.max_smooth_iters:
                    try:
                        self.objective.smooth_p(self.smoothing_factor)
                        smooth_iters += 1
                        print("*KL converged: ", np.std(entropy), "Smoothing distribution.")
                        set_value = self.objective.run(None) 
                        entropy.clear()
                    
                        # Initialize Priority Queue
                        upper_bounds_ = PriorityQueue()
                        while not upper_bounds.empty():
                            alpha_ub, idx_ub = upper_bounds.get(False)
                            new_set_value = self.objective.run(idx_ub)
                            gain_ub = (new_set_value - set_value)
                            if self.cost_select:
                                gain_ub /= self.cost(self.test_wordlist[idx])
                            upper_bounds_.put((-gain_ub, idx_ub))

                        upper_bounds = upper_bounds_
                        continue

                    except AttributeError:
                        pass


            else:
                upper_bounds.put((-gain, idx))
                
        return [self.test_wordlist[idx] for idx in S]



class RandomSubset(object):
    def __init__(self, objective, budget, wordlist, cost=lambda x: 1, cost_select=False):
        self.budget = budget
        self.test_wordlist = wordlist
        self.cost = cost
    
    
    def run(self):
        ranked_words = []
        V_minus_S = [w for w in self.test_wordlist if self.cost(w) <= self.budget]
        total_cost = 0.0
        
        while total_cost < self.budget and len(V_minus_S) > 0:
            w_star = V_minus_S.pop(np.random.randint(len(V_minus_S)))
            ranked_words.append(w_star)
            total_cost += self.cost(w_star)
            print(w_star.encode('utf-8'), "Consumed: ", total_cost, "/", self.budget, ": + ", self.cost(w_star), end=" : ")
            
            remaining_budget = self.budget - total_cost
            V_minus_S = [w for w in V_minus_S if self.cost(w) <= remaining_budget]
            print("Remaining Candidates: ", len(V_minus_S))
            
        return ranked_words
    
    def run_lazy(self):
        return self.run()
