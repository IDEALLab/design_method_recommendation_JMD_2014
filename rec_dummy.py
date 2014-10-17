'''
    Some Baseline Recommendation models used for comparison in the paper experiments
    Written by Mark Fuge
'''

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class Popularity(BaseEstimator, ClassifierMixin):
    '''
    Dummy recommender which just recommends methods in order of popularity
    '''
    def __init__(self):
        self.method_ranking=None
    
    def fit(self,X,y):
        self.method_prob = sum(y)/len(y)  
        
    def predict_proba(self,X):
        try:
            n,m = X.shape
            return np.tile([1-self.method_prob,self.method_prob],[n,1])
        except ValueError:
            print 'Attempted to predict before fitting data'
            
class RandomClassifier(BaseEstimator, ClassifierMixin):
    '''
    Dummy recommender which just recommends methods randomly
    '''
    def __init__(self):
        self.method_ranking=None
    
    def fit(self,X,y):
        pass
        
    def predict_proba(self,X):
        n,m = X.shape
        probs = np.random.rand(n,1)
        return np.hstack([1-probs,probs])