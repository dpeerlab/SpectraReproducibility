from sklearn.decomposition import NMF
from EM_SPADE import EMSPADE
from opt_einsum import contract
import scanpy as sc
import itertools
from scipy.stats import spearmanr
import numpy as np 
import seaborn as sns
from ZIFA import ZIFA
import pandas as pd
from pyspade_global import *
import matplotlib.pyplot as plt
import netNMFsc
from sklearn.metrics import roc_curve, auc 
from schpf import scHPF, run_trials, run_trials_pool
from schpf import load_model, save_model
import slalom
from slalom import plotFactors, plotRelevance, saveFA, dumpFA
from util import *
import time
model_kwargs = dict(a=0.3, c=0.3)
def simulate_base_data(N,k,p, scale = 1):
    theta_star = np.random.exponential(scale = scale, size = (p,k))
    theta_star[theta_star < 2] = 0 
    cov = np.eye(k) 
    lst = []
    for i in range(N):
        a = np.exp(np.random.multivariate_normal(np.zeros(k),cov))
        lst.append(a)
    A_star = np.array(lst)
    A_star[A_star < 1] = 0
    global_mean = contract('ik,jk->ij',A_star,theta_star)
    data = np.random.poisson(global_mean)
    return(data,A_star,theta_star)

def create_pathways(n_control_pathways,n_active_pathways, gene_set_size, p, N, overlap,signal_strength):
    ct = 0
    ct2 = 0
    base = np.zeros((n_active_pathways,p))
    lst = []
    gene_sets = []
    for i in range(n_active_pathways):
        base[ct2, ct:ct + gene_set_size] = np.random.exponential(scale = signal_strength, size = gene_set_size)
        gene_sets.append(list(range(ct,ct + gene_set_size)))
        ct = ct + int((1 - overlap)*gene_set_size)
        ct2 = ct2 + 1
    for i in range(n_control_pathways):
        gene_sets.append(list(range(ct,ct + gene_set_size)))
        ct = ct + int((1 - overlap)*gene_set_size)
        ct2 = ct2 + 1
        
    cov = np.eye(n_active_pathways) 
    lst = []
    for i in range(N):
        a = np.exp(np.random.multivariate_normal(np.zeros(n_active_pathways),cov))
        lst.append(a)
    A_star = np.array(lst)
    A_star[A_star < 1] = 0
    global_mean = contract('ik,kj->ij',A_star,base)
    data = np.random.poisson(global_mean)
    return data,base, A_star,gene_sets

def remove(lst,idx):
    lst_return = []
    for i in range(len(lst)):
        if i not in idx:
            lst_return.append(lst[i])
    return lst_return
def noisy_gene_sets(gene_sets,p,FNR,FPR):
    noisy = []
    q = len(gene_sets[0])
    for gene_set in gene_sets:
        
        to_drop = np.random.permutation(len(gene_set))[:int(FNR*q)]
        new_gene_set = remove(gene_set, to_drop)
        new_gene_set = new_gene_set + list(np.random.permutation(p)[:int(FPR*q)])
        noisy.append(new_gene_set)
    return noisy
 
def create_mask(input_lst, G_input, D):
    input_mask = np.zeros((G_input, D))
    for i in range(len(input_lst)):
        for input_ in input_lst[i]:
            input_mask[i,input_]  = 1.0 
    return input_mask
    
def initialize_total(model,gene_sets,val):    
        """
        form of gene_sets:
        
        cell_type (inc. global) : set of sets of idxs
        """
        
        for ct in model.cell_types:
            assert(model.L[ct] >= len(gene_sets[ct]))
            count = 0
            if model.L[ct] > 0:
                if len(model.adj_matrix[ct]) > 0:
                    for gene_set in gene_sets[ct]:
                        model.theta[ct].data[:,count][gene_set] = val
                        count = count + 1
                    #for i in range(self.L[ct]):
                    #    self.eta[ct].data[i,-1] = -val
                    #    self.eta[ct].data[-1,i] = -val
                    #self.theta[ct].data[:,-1][self.adj_matrix[ct].sum(axis = 1) == 0] = val
                    #self.theta[ct].data[:,-1][self.adj_matrix[ct].sum(axis = 1) != 0] = -val

        assert(model.L["global"] >= len(gene_sets["global"]))
        count = 0
        for gene_set in gene_sets["global"]:
            model.theta["global"].data[:,count][gene_set] = val
            count = count + 1
        #for i in range(self.L["global"]):
        #    self.eta["global"].data[i,-1] = -val
        #    self.eta["global"].data[-1,i] = -val
        #self.theta["global"].data[:,-1][self.adj_matrix["global"].sum(axis = 1) == 0] = val
        #self.theta["global"].data[:,-1][self.adj_matrix["global"].sum(axis = 1) != 0] = -val
        
#base parameters
n_active_pathways = 5
n_control_pathways = 5
gene_set_size = 20
p = 2000
N = 300 
k = 5
overlap = 0.3
signal_strength = 15
lam = 500/((gene_set_size)*(gene_set_size - 1))
gene_set_FPR = 0.2
gene_set_FNR = 0.2





models = []
time_value = []
param_vals = []
for n_active_pathways in [100,50,20,10]:
    for trial in range(3):
        np.random.seed(trial)
        lam = 500/((gene_set_size)*(gene_set_size - 1))
        data2, A_star2, theta_star2 = simulate_base_data(N= N,k = k,p = p)
        data,base, A_star,gene_sets = create_pathways(n_control_pathways,n_active_pathways=n_active_pathways, gene_set_size=gene_set_size,p =p,N=N,overlap = overlap, signal_strength = signal_strength)
        noisy_gs = noisy_gene_sets(gene_sets, p, gene_set_FNR,gene_set_FPR) 
        X = data + data2
        X = X + np.random.exponential(scale = 0.1, size = X.shape)
       
        
        adj_matrix = np.zeros((p,p))
        for gene_set in noisy_gs:
            for i in gene_set:
                for j in gene_set:
                    if i!=j:
                        adj_matrix[i,j] = 1
        
        I = create_mask(noisy_gs, G_input = len(noisy_gs), D= X.shape[1]).T.astype(int)
        terms = np.array([str(i) for i in range(I.shape[1])])
        start = time.time()
        FA = slalom.initFA(Y = X.astype(float), I = I, terms = terms,noise='gauss', nHidden=5, nHiddenSparse=0,do_preTrain=False, minGenes = 1, pruneGenes = False)
        FA.train()
        end = time.time()
        time_value.append(start - end)
        models.append("slalom")
        param_vals.append(n_active_pathways)
        adict = {"global":adj_matrix, "ct": []}
        L = {"global": k + n_active_pathways + n_control_pathways, "ct": 0}
        labels = np.array(N*["ct"])
        start = time.time()
        gdspade = SPADE(X = X,L = L,labels = labels,adj_matrix = adict,lam = 0.0001,kappa = 0.3,rho = 0.4,delta = 0.2)
        initialize_total(gdspade,{"global": noisy_gs, "ct":[]}, 25.0)
        train(gdspade)
        end = time.time()
        time_value.append(start - end)
        models.append("spectra")
        param_vals.append(n_active_pathways)
        dimensions = k + n_active_pathways + n_control_pathways
        n_inits = 1
        max_iters = 100000
        n_jobs = 1
        # Gradient descent operator (generally performs significantly faster than multiplicative update)
        start = time.time()
        operator = netNMFsc.netNMFGD(d=dimensions, n_inits=n_inits, max_iter=max_iters, n_jobs=n_jobs)
        operator.N = adj_matrix
        operator.X = X.T
        operator.genes = [str(i) for i in range(p)]
        W = operator.fit_transform()
        end = time.time()
        time_value.append(start - end)
        models.append("net-nmf")
        param_vals.append(n_active_pathways)
    
df = pd.DataFrame()
df["model"] = models
df["time(s)"] = time_value
df["gene_sets"] = param_vals
df["time(s)"] = -1*df["time(s)"]
df2 = df[df.model == "spectra"]
sns.lineplot(
    data=df,
    x="gene_sets", y="time(s)", hue="model", style="model",
    markers=True, dashes=False
)
plt.savefig("simulations/time_genesets.svg")