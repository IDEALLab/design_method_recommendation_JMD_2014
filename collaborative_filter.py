'''
    Primary Collaborative Filtering Code used for method recommendation
    Written by Mark Fuge and Bud Peters
    Heavily based on the 'Scalable Collaborative Filtering' code written by Mark Fuge at:
    https://github.com/markfuge/scalable-collaborative-filtering
    
    For more information on the project visit:
    http://ideal.umd.edu/projects/design_method_recommendation.html
'''

import numpy as np
import math
import csv
import math
from random import shuffle
from sklearn import cross_validation
from sklearn.base import BaseEstimator
import pdb

remap = lambda x,min1,max1,min2,max2: (((x - min1) * (max2 - min2)) / (max1 - min1)) + min2

def y_hat(mc, bc, bm, nu_c, nu_m,nu_t=None):
    ''' Standard logistic regression '''
    f_y = score(mc, bc, bm, nu_c, nu_m,nu_t=None)    
    return (1 / (1 + math.exp(-f_y)))

def score(mc, bc, bm, nu_c, nu_m,nu_t=None):
    f_y = (mc + bc + bm + np.dot(nu_c, nu_m))
    if nu_t is not None:
        # Add the category adjustment
        f_y += np.dot(nu_c, nu_t)
    f_y = min(max(f_y,-100),100)   # To prevent extreme range overuns in the exponent
    return f_y

def logistic_loss(y_true,mc, bc, bm, nu_c, nu_m,nu_t=None):
    ''' Logistic Loss: ln(1+exp(-y*score))
    '''
    mt = y_true*score(mc, bc, bm, nu_c, nu_m,nu_t=None)
    return math.log(1+math.exp(-mt))

def adjust_eta(alpha, beta):
    return lambda t: 1/(math.sqrt(alpha+beta*t))
    
def calc_eta(t,alpha, beta):
    return 1/(math.sqrt(alpha+beta*t))

def split_data(filename, size):
    reader = csv.reader(open(filename, 'rb'), delimiter=',')
    data = []
    for row in reader:
        data.append(row)
    return cross_validation.train_test_split(data, test_size=size, random_state=0)
    
def get_or_init(dictionary, id, init_function):
    '''
    Looks up a value in a dictionary if the key exists, otherwise 
    creates a new random latent factor vector and assigns
    it to the desired dictionary.
    '''
    if not id in dictionary:
        # Need to initialize a random vector for a new case
        dictionary[id] = init_function
    return dictionary[id]

def init_latent_factor_vector(dimensions):
    return 5*(np.ones(dimensions))
    
def preprocess_recommendations(Y):
    ''' Takes in a binary recommendation matrix and outputs an array of tuples
        that can be used by the stochastic gradient descent algorithm
    '''
    data = []
    for case_num,case in enumerate(Y.tolist()):
        for method_num,method_used in enumerate(case):
            method_used = method_used if method_used == 1 else -1
            data.append((case_num,method_num,method_used))
    return data

class CollaborativeFilter(BaseEstimator):
    '''
    A collaborative filter object for storing and operating over the various
    parameters and the loss function.
    '''
    def __init__(self, categories=None, alpha=1, beta=0.005,
                 lambda_nu=1.0, lambda_b=1.0, num_latent=10):
        '''
        Initializes the collaborative filtering model.
        '''
        self.num_latent = num_latent
        self.categories = categories
        
        # Initialize the learning rate for stochastic gradient descent
        self.initial_eta = 10
        self.alpha = alpha
        self.beta = beta
        self.lambda_nu = lambda_nu
        self.lambda_b = lambda_b
        self.reset_params()
    
    def reset_params(self):
        self.bm = {}
        self.bc = {}
        self.mc = 0.10
        
        # Initialize the latent factor space for cases and methods
        # Random initialization
        self.nu_c = {}   # Cases
        self.nu_m = {}   # Methods
        self.nu_t = {}   # Categories
        self.iteration = 0
    
    def set_params(self,new_params):
        for key,val in new_params.iteritems():
            setattr(self,key,val)
        
    def fit(self,X,y):
        # Get the data into a sequential form that can be processed by SGD
        self.reset_params()
        # Do several passes through dataset for Stochastic Gradient Descent
        num_passes = 20 # Number of repeated passes through the data
        for iteration in range(num_passes):
            self.single_pass_update(X,y)
            
    def single_pass_update(self,X,recommendations):
        ''' Does a single Stochastic Gradient Descent pass through the data '''
        total_loss, average_loss = 0, 0
        for (case_id,method_id),rec in zip(X,recommendations):
            loss = self.update(int(case_id), int(method_id), int(rec))
            total_loss += loss
            average_loss += loss
        #print 'Average Training Loss:' + str(total_loss/len(recommendations))
        
    def update(self, case_id, method_id, rec):
        # Determine the descent step size
        self.iteration += 1
        eta = calc_eta(self.iteration,self.alpha,self.beta)
        nu_c, bc = self.get_case(case_id)
        
        if type(self.categories)=='numpy.ndarray':
            category_list = self.categories[case_id]
            nu_t = self.sum_categories(category_list)
        else:
            nu_t = None
        
        nu_m, bm = self.get_method(method_id)
        
        mc = self.mc
        discount = 1-self.lambda_nu*eta
        
        mt = rec*score(mc,bc,bm,nu_c,nu_m,nu_t)
        # from the Logistic Loss derivative - error term
        err = lambda y,f: -y*(1/(1+math.exp(y*f)))
        
        # Adjust the global term
        mc = mc - eta*err(rec,score(mc,bc,bm,nu_c,nu_m,nu_t))
        
        # Adjust the independent terms
        discount = 1-self.lambda_b*eta
        bc = discount*bc - eta*err(rec,score(mc,bc,bm,nu_c,nu_m,nu_t))
        bm = discount*bm - eta*err(rec,score(mc,bc,bm,nu_c,nu_m,nu_t))
        
        # Adjust the latent factors
        discount = 1-self.lambda_nu*eta
        ## Methods
        nu_m = discount*nu_m - eta*nu_c*err(rec,score(mc,bc,bm,nu_c,nu_m,nu_t))
        ## Categories
        if type(self.categories)=='numpy.ndarray' and len(category_list)>0:
            error = err(rec,score(mc,bc,bm,nu_c,nu_m,nu_t))
            nu_t=0
            for category_name in category_list:
                # Regularize the category
                self.nu_t[category_name] = discount*self.nu_t[category_name]
                # Then subtract off the prediction error amounts
                self.nu_t[category_name] -= eta*nu_c*error
                nu_t += self.nu_t[category_name]  
        ## Cases
        nu_c = discount*nu_c
        nu_c -= eta*nu_m*err(rec,score(mc,bc,bm,nu_c,nu_m,nu_t))
        if nu_t is not None:
            nu_c -= eta*nu_t*err(rec,score(mc,bc,bm,nu_c,nu_m,nu_t))
        # Now assign every back for the next round
        self.mc = mc
        self.bm[method_id] = bm
        self.bc[case_id] = bc
        self.nu_c[case_id] = nu_c
        self.nu_m[method_id] = nu_m
        return logistic_loss(rec,mc, bc, bm, nu_c, nu_m, nu_t)
        
    def decision_function(self,X):
        ''' Uses partial information about the test matrix to predict 
            other methods used.
        '''
        Y_hat=[]
        for case_id,method_id in X:
            Y_hat.append(self.predict(int(case_id),int(method_id)))
        return np.array(Y_hat)
        
    def predict_proba(self,X):
        return self.decision_function(X)
        
    def predict(self, case_id, method_id):
        ''' Predicts the method likelihood from a given case and method '''
        nu_c, bc = self.get_case(case_id)
        nu_m, bm = self.get_method(method_id)
        if type(self.categories)=='numpy.ndarray':
            nu_t = self.sum_categories(self.categories[case_id])
        else:
            nu_t = None
        pred = y_hat(self.mc,bc,bm,nu_c,nu_m,nu_t)
        return pred
        
    def get(self, dictionary, id, init_function):
        '''
        Looks up a value in a dictionary if the key exists, otherwise 
        creates a new random latent factor vector and assigns
        it to the desired dictionary.
        '''
        if not id in dictionary:
            # Need to initialize a random vector for a new case
            dictionary[id] = init_function
        return dictionary[id]
        
    def get_vectors(self,id,nu,b):
        '''
        Returns the nu and b vectors for a specific id.
        '''
        return (get_or_init(nu, id, init_latent_factor_vector(self.num_latent)),
                get_or_init( b, id, 1*(np.random.rand()-0.5)))
                
    def get_case(self, case_id):
        return self.get_vectors(case_id,self.nu_c,self.bc)
    
    def get_method(self, method_id):
        return self.get_vectors(method_id,self.nu_m,self.bm)
    
    def get_category(self, category_id):
        return get_or_init(self.nu_t, category_id, init_latent_factor_vector(self.num_latent))
                
    def sum_categories(self,category_ids):
        '''Returns the sum of category vectors for a set of categories'''
        return sum(self.get_category_list(category_ids))

    def get_category_list(self,category_ids):
        '''Returns the sum of category vectors for a set of categories'''
        return [self.get_category(category_id) for category_id in category_ids]