'''
  Some helper functions to assist in the plotting and wrangling of data
'''

import numpy as np
import matplotlib.pylab as plt
from matplotlib.ticker import MaxNLocator,AutoLocator

almost_black = '#262626'

def is_homogeneous(l):
    ''' Checks to see if a list has all identical entries  
    '''
    for i in range(len(l) - 1):
        if l[i] != l[i + 1]:
            return False
    return True

def min_member_class_ok(l):
    ''' Finds the member label with the lowest occurrence rate  
    '''
    return (get_min_class(l)>1 and not is_homogeneous(l))
    
def get_min_class(l):
    ''' Returns the cardinality of the label set. At least two are needed.
    '''
    classes = {}
    for i in l:
        if i in classes: 
            classes[i] += 1
        else:
            classes[i] = 1
    min = len(l)
    for j in classes.iteritems():
        if j[1] < min:
            min = j[1]
    return min
    
def setfont():
    ''' Sets some fonts for plotting the figures
    '''
    font = {'family' : 'serif',
            'weight' : 'normal',
            'size' : 40}
    plt.matplotlib.rc('font', **font)
    

def setup_plots():
    setfont()
    # Close any currently open MatPlotLib figures
    plt.close('all')
    PR_fig = plt.figure('MAP',figsize=(15,8),dpi=80)
    #mng = plt.get_current_fig_manager()
    #mng.resize(*mng.window.maxsize())
    plt.title('Precision-Recall Performance')
    plt.hold(True)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim((0,1.05))
    plt.xlim((0,1.05))
    return PR_fig

def fix_legend(ax=None,**kwargs):
    '''Applies nice coloring to legend'''
    if not ax:
        ax = plt.gca()
    light_grey = np.array([float(248)/float(255)]*3)
    legend = ax.legend(frameon=True,fontsize=16,**kwargs)
    ltext = ax.get_legend().get_texts()
    for lt in ltext:
        plt.setp(lt, color = almost_black)
    rect = legend.get_frame()
    rect.set_facecolor(light_grey)
    rect.set_linewidth(0.0)
    # Change the legend label colors to almost black, too
    texts = legend.texts
    for t in texts:
        t.set_color(almost_black)
    
def fix_axes(ax=None):
    '''
      Removes top and left boxes
      Lightens text
    '''
    if not ax:
        ax = plt.gca()
    # Remove top and right axes lines ("spines")
    spines_to_remove = ['top', 'right']
    for spine in spines_to_remove:
        ax.spines[spine].set_visible(False)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5,prune='lower'))
    # Change the labels to the off-black
    ax.xaxis.label.set_color(almost_black)
    ax.yaxis.label.set_color(almost_black)