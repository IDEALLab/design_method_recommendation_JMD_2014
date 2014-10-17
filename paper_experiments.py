'''
    Code for method recommendation experiments for HCD Connect cases
    Written by Mark Fuge and Bud Peters
    For more information on the project visit:
    http://ideal.umd.edu/projects/design_method_recommendation.html
    
    To reproduce the experimental results just run:
    python paper_experiments.py
    
    This experiment code is what was used to the produce the results in
    Mark Fuge, Bud Peters, Alice Agogino,  "Machine learning algorithms for recommending design methods." Journal of Mechanical Design 136 (10)
    @article{fugeHCD2014JMD,
      author = {Fuge, Mark and Peters, Bud and Agogino, Alice},
      day = {18},
      doi = {10.1115/1.4028102},
      issn = {1050-0472},
      journal = {Journal of Mechanical Design},
      month = aug,
      number = {10},
      pages = {101103+},
      title = {Machine Learning Algorithms for Recommending Design Methods},
      url = {http://dx.doi.org/10.1115/1.4028102},
      volume = {136},
      year = {2014}
    }
'''

from time import time
import csv
import re
import os
from operator import itemgetter
import numpy as np
import cPickle as pickle
import matplotlib.pylab as plt
from sklearn import svm, cross_validation, tree, ensemble
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import precision_recall_curve, average_precision_score, adjusted_mutual_info_score, adjusted_rand_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.grid_search import RandomizedSearchCV
from sklearn.cluster import SpectralClustering
from sklearn.covariance import GraphLassoCV, empirical_covariance
from sklearn.utils import resample
from scipy.stats import gamma,randint
from scipy.stats import scoreatpercentile as percentile
from collaborative_filter import *
from rec_utils import *
from rec_dummy import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD
import brewer2mpl
import matplotlib.colors
paired = brewer2mpl.get_map('Paired', 'qualitative', 10).mpl_colors
PuBuGn4 = brewer2mpl.get_map('PuBuGn', 'sequential', 5).mpl_colors

def load_hcd_cases(data_path):
    ''' Loads case study data for stories, methods, and cases
        Assumes that methods and stories are correctly ordered by line number
    '''
    # Load Stories into an indexed array table
    story_csv = csv.reader(open(data_path+'stories.csv'),delimiter = '|')
    method_csv = csv.reader(open(data_path+'methods.csv'),delimiter = '|')
    case_csv = csv.reader(open(data_path+'cases.csv'),delimiter = '|')
    
    stories=[]
    case_categories=[]
    for story in story_csv:
        stories.append((story[1].lower(),story[2].lower()))
        # Just Focus Area
        case_categories.append(story[8].lower().split(';'))
        # Focus Area + User
        #uid = ['IDEO' if story[5]=='IDEO.org' else 'noIDEO' ]
        #case_categories.append(story[8].lower().split(';')+uid)
        
    methods={}
    for method in method_csv:
        methods[int(method[0])]=[method[1],method[2]]
    methods = methods.values()
        
    cases=np.zeros((len(stories),len(methods))) 
    for story_id,method_id in case_csv:
        cases[int(story_id)][int(method_id)]=1
    
    # Now remove invalid cases:
    ft=np.array([True if len(s[1])>6 else False for s in stories])
    
    
    return np.array(stories)[ft],methods,np.array(cases)[ft],np.array(case_categories)[ft]

def get_case_mutual_information(case_binary_matrix):
    '''
    Calculates the mutual information between methods given a binary case matrix
    Output shape of the matrix should be symmetric num_methods by num_methods
    '''
    num_cases,num_methods = case_binary_matrix.shape
    MI = np.zeros(shape=(num_methods,num_methods))
    for i in range(num_methods):
        for j in range(i,num_methods):
            c1 = case_binary_matrix[:,i]
            c2 = case_binary_matrix[:,j]
            MI[i][j] = adjusted_mutual_info_score(c1,c2)
            MI[j][i] = MI[i][j]
    return MI
    
# Utility function to report best scores
# from http://scikit-learn.org/stable/auto_examples/randomized_search.html
def opt_report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

def rebalance_cases(cases):
    ''' Balances the positive and negative case results '''        
    ind = np.array([c[2]>0 for c in cases])
    pos = cases[ind]
    neg = cases[~ind]
    np.random.shuffle(pos)
    np.random.shuffle(neg)
    if len(pos)<len(neg):
        neg = neg[:len(pos)]
    else:
        pos = pos[:len(neg)]
    cases=np.vstack([pos,neg])
    np.random.shuffle(cases)
    return cases

# List some options for hyperparamer optimization
# Helpful little snippet for exploring gamma PDF functions
#x=np.linspace(0,0.02);h = plt.plot(x, gamma(13,scale=0.002).pdf(x)); plt.show()
hyper_params = {'CollaborativeFilter': 
                    {'alpha':  gamma(27,scale=0.04),#gamma(20,scale=0.10),#gamma(11,scale=0.15),#,
                      'beta' : gamma(13,scale=0.002),#gamma(20,scale=0.001),#,
                      'lambda_nu' : gamma(58,scale=0.025),#gamma(58,scale=0.025),#gamma(11,scale=0.15),#gamma(3,scale=0.4),
                      'lambda_b' : gamma(15,scale=0.008),#,#gamma(6,scale=0.05),#gamma(3,scale=0.4),
                      'num_latent' : randint(2,15)#[5,10,20,40]
                    },
                'RandomForestClassifier':
                    { 'estimator__n_estimators':  [5,10,20,40,80],
                      'estimator__criterion' : ['gini','entropy'],
                      'estimator__min_samples_split' : randint(1,10),
                      'estimator__min_samples_leaf' : randint(1,10),
                      'estimator__bootstrap': [True, False]
                    },
                'SVC':
                    { 'estimator__C':  gamma(3,scale=1.5),
                      'estimator__gamma' : gamma(3,scale=.15)
                    },   
                'LogisticRegression':
                    { 'estimator__penalty': ['l2'],#['l1','l2'],
                      'estimator__C' : gamma(3,scale=1.5),
                      'estimator__fit_intercept' : [True],#[True, False],
                      'estimator__intercept_scaling' : gamma(3,scale=1.5)
                    },
                'BernoulliNB':
                    { 'estimator__alpha':  gamma(30,scale=30)#gamma(3,scale=3)
                    },
                'RandomClassifier':{ },
                'Popularity':{ }
                }
# Storage of the optimal parameters found during previous runs of the algorithms
# This is only to speed up testing and evaluation, as well as to best allow
# others to replicate the conditions we used in the paper
optimal_params = {'CollaborativeFilter': 
                    {'alpha':  1.0,
                      'beta' : 0.021,
                      'lambda_nu' : 1.5,
                      'lambda_b' : 0.06,
                      'num_latent' : 11
                    },
                'RandomForestClassifier':
                    { 'estimator__n_estimators':  80,
                      'estimator__criterion' : 'entropy',
                      'estimator__min_samples_split' : 7,
                      'estimator__min_samples_leaf' : 5,
                      'estimator__bootstrap': False
                    },
                'SVC':
                    { 'estimator__C':  5.0,
                      'estimator__gamma' : 0.8
                    },   
                'LogisticRegression':
                    { 'estimator__penalty': 'l2',
                      'estimator__C' : .45,
                      'estimator__fit_intercept' : True,
                      'estimator__intercept_scaling' : 7.5
                    },
                'BernoulliNB':
                    { 'estimator__alpha':  100000 #4
                    },
                'RandomClassifier':{ },
                'Popularity':{ }
                }
                    
def run_classifier(clf,features,cases,bottom_inds,optimize_hyperparams=False):
    clf_name = clf.__class__.__name__
    cases = np.array(cases)
    # Set up the cross_validation study
    if clf_name == 'CollaborativeFilter':
        cases = np.array(preprocess_recommendations(cases))
        cy = [c[2] for c in cases]
        cases = np.array(cases)
        m_ind=cases[:,1]
    else:
        cy = cases[:,0]        
        
    # Pre-Run Hyperparameter Optimization
    if optimize_hyperparams:
        param_dist = hyper_params[clf_name]
    else:
        opt_param_dist = optimal_params[clf_name]
    num_iterations = 1 if optimize_hyperparams else 100
    shuffle = cross_validation.StratifiedShuffleSplit(y=cy,
                                                      n_iter=num_iterations, 
                                                      test_size=0.1,
                                                      random_state=None)
    scores =[]; Y_pred = []; Y_true = []; m_test_inds=[]
    # Run study
    for i,(train_index, test_index) in enumerate(shuffle):
        # Separate training/test set
        if (i%10)==0:
            print '  CV#%d of %d...'%(i,num_iterations)
        Y_train, Y_test = (cases[train_index],cases[test_index])
        
        # Fit and predict using the models
        if clf_name == 'CollaborativeFilter':
            # Split the training data into X and y vectors
            #Y_train = rebalance_cases(Y_train)
            X_train = Y_train[:,:-1]
            Y_train = Y_train[:,-1]
            X_test = Y_test[:,:-1]
            Y_test = Y_test[:,-1]
            m_test_ind = m_ind[test_index]
            m_test_inds.append(m_test_ind)
            
            if optimize_hyperparams:
                # Run Parameter Search
                n_iter_search = 2
                random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                                   n_iter=n_iter_search,
                                                   scoring='average_precision',
                                                   n_jobs=3,
                                                   refit=True,
                                                   cv=4,
                                                   verbose=1
                                                   )
                start = time()
                random_search.fit(X_train,Y_train)
                print("RandomizedSearchCV took %.2f minutes for %d candidates"
                      " parameter settings." % ((time() - start)/60.0, n_iter_search))
                opt_report(random_search.grid_scores_,n_top=10)
                Y_hat=random_search.best_estimator_.predict_proba(X_test)
            else:
                clf.set_params(opt_param_dist)
                clf.fit(X_train,Y_train)
                Y_hat=clf.predict_proba(X_test)
        else:
            X_train, X_test = (features[train_index],features[test_index])
            ovr = OneVsRestClassifier(clf)
            if optimize_hyperparams:
                n_iter_search = 400
                random_search = RandomizedSearchCV(ovr, param_distributions=param_dist,
                                                   n_iter=n_iter_search,
                                                   # Average precisions scoring
                                                   # doesn't seem to work in
                                                   # multi-label case
                                                   #scoring='average_precision',
                                                   scoring='log_loss',
                                                   n_jobs=3,
                                                   refit=True,
                                                   cv=4,
                                                   verbose=1
                                                   )
                start = time()
                random_search.fit(X_train,Y_train)
                print("RandomizedSearchCV took %.2f minutes for %d candidates"
                      " parameter settings." % ((time() - start)/60.0, n_iter_search))
                opt_report(random_search.grid_scores_,n_top=10)
                #clf = random_search.best_estimator_
                Y_hat=random_search.best_estimator_.predict_proba(X_test)
            else:
                ovr.set_params(**opt_param_dist)
                ovr.fit(X_train,Y_train)
                Y_hat = ovr.predict_proba(X_test)
            #Y_hat = clf.predict_proba(X_test)
        # Collect the results
        Y_pred.append(Y_hat)
        Y_true.append(Y_test)
    Y_true=np.vstack(Y_true)
    Y_pred=np.vstack(Y_pred)
    
    
    # Now do the overall AUC scoring
    print 'Generating bootstrap samples...'
    A=np.vstack([Y_true.flatten(),Y_pred.flatten()])
    A=A.transpose()
    auc_scores=[]
    for j in range(1000):
        B=resample(A)
        auc_scores.append(average_precision_score(B[:,0], B[:,1]))
    auc_scores=np.array(auc_scores)
    # Now just test PR on the k least popular methods
    if re.search('CollaborativeFilter',clf_name):
        m_test_inds=np.vstack(m_test_inds)
        m_test_ind = m_test_inds.flatten()
        ix=np.in1d(m_test_ind.ravel(), bottom_inds).reshape(m_test_ind.shape)
        A = np.vstack([Y_true.flatten()[ix],Y_pred.flatten()[ix]])
    else:
        A=np.vstack([Y_true[:,bottom_inds].flatten(),Y_pred[:,bottom_inds].flatten()])
    A=A.transpose()
    bottom_k_auc_scores=[]
    for j in range(1000):
        B=resample(A)
        bottom_k_auc_scores.append(average_precision_score(B[:,0], B[:,1]))
    bottom_k_auc_scores=np.array(bottom_k_auc_scores)
    
    return Y_pred,Y_true, auc_scores, bottom_k_auc_scores

def run_clustering(methods, cases):
    true_method_groups = [m[1] for m in methods]
    edge_model = GraphLassoCV(alphas=4, n_refinements=5, n_jobs=3, max_iter=100)
    edge_model.fit(cases)
    CV = edge_model.covariance_
    
    num_clusters=3
    spectral = SpectralClustering(n_clusters=num_clusters,affinity='precomputed') 
    spectral.fit(np.asarray(CV))
    spec_sort=np.argsort(spectral.labels_)
    
    for i,m in enumerate(methods):
        print "%s:%d\t%s"%(m[1],spectral.labels_[i],m[0])
    print "Adj. Rand Score: %f"%adjusted_rand_score(spectral.labels_,true_method_groups)

def run_method_usage(methods,cases):
    methods = [m[0] for m in methods]
    # Bootstrap the percentage error bars:
    percents =[]
    for i in range(10000):
        nc = resample(cases)
        percents.append(100*np.sum(nc,axis=0)/len(nc))
    percents=np.array(percents)
    mean_percents = np.mean(percents,axis=0)
    std_percents = np.std(percents,axis=0)*1.96
    inds=np.argsort(mean_percents).tolist()
    inds.reverse()
    avg_usage = np.mean(mean_percents)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x=np.arange(len(methods))
    ax.plot(x,[avg_usage]*len(methods),'-',color='0.25',lw=1,alpha=0.2)
    ax.bar(x, mean_percents[inds], 0.6, color=paired[0],linewidth=0,
           yerr=std_percents[inds],ecolor=paired[1])
    #ax.set_title('Method Occurrence')
    ax.set_ylabel('Occurrence %',fontsize=30)
    ax.set_xlabel('Method',fontsize=30)
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels(np.array(methods)[inds],fontsize=8)
    fig.autofmt_xdate()
    fix_axes()
    plt.tight_layout()
    fig.savefig(figure_path+'method_occurrence.pdf', bbox_inches=0)
    fig.show()
    return inds,mean_percents[inds]
    
# main script generates results and plots
if __name__ == "__main__":
    print_output = True
    plot_output = True
    k=10    # the number of least popular methods to include in the reduced PR part
    data_path='../data/'
    results_path='results/'
    figure_path='figures/'
    # Check if needed directories exist, if not, create it
    for path in [data_path, results_path, figure_path]:
        if not os.path.exists(path):
            os.makedirs(path)
    
    # Load data
    print 'Loading data...'
    story_text, methods, cases,case_categories = load_hcd_cases(data_path) 
    dataset=[s[1] for s in story_text]
    vectorizer = TfidfVectorizer(max_df=0.5,stop_words='english')
    X = vectorizer.fit_transform(dataset)
    lsa = TruncatedSVD(n_components=50,algorithm='arpack')
    X = lsa.fit_transform(X)
    classifier_features = Normalizer(copy=False).fit_transform(X)
    
    # Clustering Part
    print 'Running Clustering...'
    run_clustering(methods, cases)
    
    # Method Usage:
    print 'Running Method Usage...'
    inds,method_freq=run_method_usage(methods, cases)
    bottom_inds=list(reversed(inds))[0:k]
    
    # Recommender System part
    if(plot_output):
        PR_fig = setup_plots()
    
    # Specify the classifiers
    clfs = [
            BernoulliNB(alpha=0.001),
            LogisticRegression(C=0.02, penalty='l1', tol=0.001),
            svm.SVC(C=1,kernel='rbf',probability=True),
            ensemble.RandomForestClassifier(),
            CollaborativeFilter(categories=False),
            Popularity(),
            RandomClassifier(),
            CollaborativeFilter(categories=case_categories)
            ]
    #plot_ops = ['k:','k--','k-','k-.','r-','b-','g-'] 
    plot_ops = [{'linewidth':8.0, 'linestyle':'-','color':PuBuGn4[1]},
                {'linewidth':8.0, 'linestyle':'-','color':PuBuGn4[2]},
                {'linewidth':8.0, 'linestyle':'-','color':PuBuGn4[3]},
                {'linewidth':8.0, 'linestyle':'-','color':PuBuGn4[4]},
                {'linewidth':3.0, 'linestyle':'-','color':PuBuGn4[4]},
                {'linewidth':3.0, 'linestyle':'-','color':PuBuGn4[3]},
                {'linewidth':3.0, 'linestyle':'-','color':PuBuGn4[2]},
                {'linewidth':3.0, 'linestyle':'-','color':PuBuGn4[1]},
               ]
    plt.rc('axes',color_cycle = paired)
    # For each classifier
    print 'Running Classifiers:'
    for i,clf in enumerate(clfs):
        clf_name=clf.__class__.__name__
        if clf_name=='CollaborativeFilter' and (clf.categories is not False):
            clf_name+='+'
        print ' '+str(clf_name)+'...'
        try:
            # If we can load the existing data file, skip this classifier
            (Y_hat,Y_true,auc_scores,bottom_k_auc_scores) = pickle.load(open(results_path+'clf_results_%s.pickle'%clf_name,'rb'))
            continue
        except IOError:
            # We haven't generated results yet, so run the classifier
            # and get the classifiers predictions
            Y_hat, Y_true, auc_scores, bottom_k_auc_scores = run_classifier(clf,classifier_features,cases, bottom_inds,
                                          optimize_hyperparams=False)
            # Save the data for next time
            print ' saving data...'
            pickle.dump((Y_hat,Y_true,auc_scores,bottom_k_auc_scores),
                        open(results_path+'clf_results_%s.pickle'%clf_name,'wb'))
        finally:
            if(plot_output):
                print ' plotting data...'
                # Plot the Precision Recall Curve
                # Scikit's Precision Recall
                p,r,thresh = precision_recall_curve(Y_true.flatten(), Y_hat.flatten())
                plt.plot(r,p,label=clf_name,**plot_ops[i])
                # Now get AUC bounds via bootstrap resampling
            print ' AUC bootstrap resampling...'
            print("[%.3f,%.3f]: %s AUC 95 bounds"%(percentile(auc_scores,2.5),percentile(auc_scores,97.5),clf_name))
            print("[%.3f,%.3f]: %s AUC 95 bounds - bottom %d methods"%(percentile(bottom_k_auc_scores,2.5),percentile(bottom_k_auc_scores,97.5),clf_name,k))
        
    # Save and display the overall figure
    if(plot_output):
        #plt.legend(loc=1,fontsize=20)
        fix_legend(handlelength=7)
        fix_axes()
        plt.ylim(0,1)
        plt.xlim(0,1)
        plt.hold(False)
        plt.tight_layout()
        PR_fig.savefig(figure_path+'precision_recall.pdf', bbox_inches=0)
        PR_fig.show()
    
    # Now print out all the results
    print "Printing PR-AUC Results for each classifier..."
    print "Classifier & All & Bottom10"
    clfs_auc=[]
    clfs_bauc=[]
    classifier_names=[]
    for clf in clfs:
        clf_name=clf.__class__.__name__
        if clf_name=='CollaborativeFilter' and (clf.categories is not False):
            clf_name+='+'
        classifier_names.append(clf_name)
        (Y_hat,Y_true,auc_scores,bottom_k_auc_scores) = pickle.load(open(results_path+'clf_results_%s.pickle'%clf_name,'rb'))
        print "%s & %.3f & %.3f"%(clf_name,np.mean(auc_scores),np.mean(bottom_k_auc_scores)) 
        clfs_auc.append(auc_scores)
        clfs_bauc.append(bottom_k_auc_scores)
    clfs_auc=np.asarray(clfs_auc)
    clfs_bauc=np.asarray(clfs_bauc)
    classifier_names = np.asarray(classifier_names)
    
    print "Plotting PR-AUC results"
    order = [7,6,0,1,2,5,3,4]
    width = 0.3 # the width of the bars
    classifier_names = classifier_names[order]
    plot_means = np.median(clfs_auc,axis=1)[order]
    plot_bmeans = np.median(clfs_bauc,axis=1)[order]
    plot_bars = np.array([abs(percentile(auc,[2.5,97.5]) - percentile(auc,50))  for auc in clfs_auc[order]]).transpose()
    plot_bbars = np.array([abs(percentile(bauc,[2.5,97.5]) - percentile(bauc,50))  for bauc in clfs_bauc[order]]).transpose()
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    plt.hold(True)
    ax.bar(np.arange(8),plot_means,width ,color=paired[1],label='All Methods',
           yerr=plot_bars, ecolor='k',linewidth=0,align='center')
    ax.bar(np.arange(8)+width, plot_bmeans, width,color=paired[0],label='Bottom 10',
           yerr=plot_bbars, ecolor='k',linewidth=0,align='center')
    ax.set_ylabel('PR AUC',fontsize=30)
    ax.tick_params(axis='y', which='major', labelsize=25)
    ax.set_xticks(np.arange(len(order))+width)
    ax.set_xticklabels(classifier_names,fontsize=20)
    fig.autofmt_xdate()
    fix_legend()
    #bbox_to_anchor=(1.0, 1.0)
    fix_axes()
    plt.tight_layout()
    fig.savefig(figure_path+'auc_compare.pdf', bbox_inches=0)
    fig.show()
