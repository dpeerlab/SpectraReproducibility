#imports
import numpy as np
import pandas as pd
from torch.distributions.beta import Beta
import os
import scipy
import seaborn as sns
import warnings
import scanpy as sc
import numpy.matlib
import numpy as np
from pyvis.network import Network
import torch
import json
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as ds
from torch.autograd import Variable
from torch import optim
from sklearn.decomposition import NMF
from opt_einsum import contract 
#from util import s_term_normal, s_term_bernoulli
from torch.distributions.normal import Normal
from torch.distributions.log_normal import LogNormal
from torch.distributions.dirichlet import Dirichlet
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import OrderedDict
import matplotlib.pyplot as plt
import warnings
import matplotlib

class SPADE(nn.Module):
    
    """
    has gene scale params
    
    
    delta parameter: testing this... 
    
    factor matrix as 
    A*Theta*diag(G + delta) with delta in (0,0.5); bounding a ratio by 1/delta or (1+delta)/delta (?)
    
    """
    def __init__(self,X,labels,adj_matrix, L = "default", weights = None, lam = 10e-4,kappa = 0.0,rho = 0.0,delta=0.1,beta = 0.0):
        super(SPADE, self).__init__()
        self.beta = beta
        self.delta = delta
        self.X = torch.Tensor(X)
        self.adj_matrix = {cell_type: torch.Tensor(adj_matrix[cell_type]) - torch.Tensor(np.diag(np.diag(adj_matrix[cell_type]))) if len(adj_matrix[cell_type]) > 0 else [] for cell_type in adj_matrix.keys()}
        adj_matrix_1m = {cell_type: 1.0 - adj_matrix[cell_type] if len(adj_matrix[cell_type]) > 0 else [] for cell_type in adj_matrix.keys()} #one adj_matrix per cell type 
        self.adj_matrix_1m = {cell_type: torch.Tensor(adj_matrix_1m[cell_type] - np.diag(np.diag(adj_matrix_1m[cell_type]))) if len(adj_matrix_1m[cell_type]) > 0 else [] for cell_type in adj_matrix_1m.keys()} #one adj_matrix per cell type 
        self.L = L
        self.p = X.shape[1]
        self.n = X.shape[0]
        self.kappa = kappa
        self.rho = rho
        self.lam = lam
        self.cell_types = np.unique(labels)
        self.labels = labels
        self.theta = nn.ParameterDict()
        self.alpha = nn.ParameterDict()
        self.eta = nn.ParameterDict()
        self.gene_scaling = nn.ParameterDict()
        self.cell_type_counts = {}
        for cell_type in self.cell_types:
            n_c = sum(labels == cell_type)
            self.cell_type_counts[cell_type] = n_c 
        if self.L == "default":
            """
            eventually estimate with BEMA
            """
            L = OrderedDict({"global":25})
            for k in self.cell_types:
                L[k] = 6
                if k == 'nan':
                    L[k] = 0
        self.L = L        
        self.theta["global"] = nn.Parameter(Normal(0.,1.).sample([self.p, self.L["global"]]))
        self.eta["global"] = nn.Parameter(Normal(0.,1.).sample([self.L["global"], self.L["global"]]))
        self.gene_scaling["global"] = nn.Parameter(Normal(0.,1.).sample([self.p]))
        for cell_type in self.cell_types:
            self.theta[cell_type] = nn.Parameter(Normal(0.,1.).sample([self.p,self.L[cell_type]]))
            self.eta[cell_type] = nn.Parameter(Normal(0.,1.).sample([self.L[cell_type], self.L[cell_type]]))
            n_c = sum(labels == cell_type)
            self.alpha[cell_type] = nn.Parameter(Normal(0.,1.).sample([n_c, self.L["global"] + self.L[cell_type]]))
            self.gene_scaling[cell_type] = nn.Parameter(Normal(0.,1.).sample([self.p]))
        if weights:
            self.weights = {cell_type: torch.Tensor(weights[cell_type]) - torch.Tensor(np.diag(np.diag(weights[cell_type]))) if len(weights[cell_type]) > 0 else [] for cell_type in weights.keys()}
        else:
            self.weights = self.adj_matrix
            
    def loss(self):
        loss = 0.0
        X = self.X
        theta_global = torch.softmax(self.theta["global"], dim = 1)
        eta_global = (self.eta["global"]).exp()/(1.0 + (self.eta["global"]).exp())
        eta_global = 0.5*(eta_global + eta_global.T)
        gene_scaling_global = self.gene_scaling["global"].exp()/(1.0 + self.gene_scaling["global"].exp())
        for cell_type in self.cell_types:
            gene_scaling_ct = self.gene_scaling[cell_type].exp()/(1.0 + self.gene_scaling[cell_type].exp())
            X_c = X[self.labels == cell_type]
            adj_matrix = self.adj_matrix[cell_type] 
            weights = self.weights[cell_type]
            adj_matrix_1m = self.adj_matrix_1m[cell_type]
            theta_ct = torch.softmax(self.theta[cell_type], dim = 1)
            eta_ct = (self.eta[cell_type]).exp()/(1.0 + (self.eta[cell_type]).exp())
            eta_ct = 0.5*(eta_ct + eta_ct.T)
            theta_global_ = contract('jk,j->jk',theta_global,gene_scaling_global + self.delta)
            theta_ct_ = contract('jk,j->jk',theta_ct,gene_scaling_ct + self.delta)
            theta = torch.cat((theta_global_, theta_ct_),1)
            alpha = torch.exp(self.alpha[cell_type])
            recon = contract('ik,jk->ij', alpha, theta) 
            term1 = -1.0*(X_c*torch.log(recon) - recon).sum()
            if len(adj_matrix) > 0:
                mat = contract('il,lj,kj->ik',theta_ct,eta_ct,theta_ct) 
                term2 = -1.0*(torch.log((1.0 -self.kappa)*mat + self.kappa)*adj_matrix*weights).sum()
                term3 = -1.0*(torch.log((1.0 -self.kappa)*(1.0 - self.rho)*(1.0 - mat) + self.rho)*(adj_matrix_1m)).sum()
            else:
                term2 = 0.0
                term3 = 0.0
            loss = loss + self.lam*term1 +(self.cell_type_counts[cell_type]/float(self.n))*(term2 + term3) 
            loss = loss - self.beta*torch.log(theta_ct + .0001).sum()
        loss = loss - self.beta*torch.log(theta_global + .0001).sum()
        #handle adj matrix for global factors
        adj_matrix = self.adj_matrix["global"] 
        adj_matrix_1m = self.adj_matrix_1m["global"]
        weights = self.weights["global"]
        if len(adj_matrix) > 0:
            mat = contract('il,lj,kj->ik',theta_global,eta_global,theta_global) 
            term2 = -1.0*(torch.log((1.0 -self.kappa)*mat + self.kappa)*adj_matrix*weights).sum()
            term3 = -1.0*(torch.log((1.0 -self.kappa)*(1.0 - self.rho)*(1.0 - mat) + self.rho)*(adj_matrix_1m)).sum()
            loss = loss + term2 + term3 
        return loss
        
        
    def initialize_total(self,gene_sets,val):    
        """
        form of gene_sets:
        
        cell_type (inc. global) : set of sets of idxs
        """
        
        for ct in self.cell_types:
            assert(self.L[ct] >= len(gene_sets[ct]))
            count = 0
            if self.L[ct] > 0:
                if len(self.adj_matrix[ct]) > 0:
                    for gene_set in gene_sets[ct]:
                        self.theta[ct].data[:,count][gene_set] = val
                        count = count + 1
                    for i in range(self.L[ct]):
                        self.eta[ct].data[i,-1] = -val
                        self.eta[ct].data[-1,i] = -val
                    self.theta[ct].data[:,-1][self.adj_matrix[ct].sum(axis = 1) == 0] = val
                    self.theta[ct].data[:,-1][self.adj_matrix[ct].sum(axis = 1) != 0] = -val

        assert(self.L["global"] >= len(gene_sets["global"]))
        count = 0
        for gene_set in gene_sets["global"]:
            self.theta["global"].data[:,count][gene_set] = val
            count = count + 1
        for i in range(self.L["global"]):
            self.eta["global"].data[i,-1] = -val
            self.eta["global"].data[-1,i] = -val
        self.theta["global"].data[:,-1][self.adj_matrix["global"].sum(axis = 1) == 0] = val
        self.theta["global"].data[:,-1][self.adj_matrix["global"].sum(axis = 1) != 0] = -val
        
        #self.eta.data[-1] = -10**2
    def initialize_few(self,val):
        """
        **gene_set should be flattened
        
        checks if gene is in a gene set for that cell type, if so initialize uniform across first k-1, 
        otherwise initialize in k'th element
        
        initialize 
        
        """
        
        for ct in self.cell_types:
            if self.L[ct] > 0:
                #full_genes = list(set([item for sublist in gene_sets[ct] for item in sublist]))
                for count in range(self.L[ct] - 1):
                    self.theta[ct].data[:,count][self.adj_matrix[ct].sum(axis = 1) == 0] = -val
                    self.theta[ct].data[:,count][self.adj_matrix[ct].sum(axis = 1) != 0] = val
                for i in range(self.L[ct]):
                    self.eta[ct].data[i,-1] = -val
                    self.eta[ct].data[-1,i] = -val
                self.theta[ct].data[:,-1][self.adj_matrix[ct].sum(axis = 1) == 0] = val
                self.theta[ct].data[:,-1][self.adj_matrix[ct].sum(axis = 1) != 0] = -val
        #full_genes = list(set([item for sublist in gene_sets["global"] for item in sublist]))
        
        for count in range(self.L["global"] - 1):
            self.theta["global"].data[:,count][self.adj_matrix["global"].sum(axis = 1) == 0] = -val
            self.theta["global"].data[:,count][self.adj_matrix["global"].sum(axis = 1) != 0] = val
        for i in range(self.L["global"]):
            self.eta["global"].data[i,-1] = -val 
            self.eta["global"].data[-1,i] = -val 
        self.theta["global"].data[:,-1][self.adj_matrix["global"].sum(axis = 1) == 0] = val
        self.theta["global"].data[:,-1][self.adj_matrix["global"].sum(axis = 1) != 0] = -val
        
def train(model, lr_schedule = [1.0,.5,.1,.01,.001,.0001], num_epochs = 10000):

    opt = torch.optim.Adam(model.parameters(), lr=lr_schedule[0])
    
    counter = 0
    last = np.inf

    for i in tqdm(range(num_epochs)):
        #print(counter)
        opt.zero_grad()
        loss = model.loss()
        loss.backward()
        opt.step()
    
        if loss.item() >= last:
            counter += 1
            if int(counter/3) >= len(lr_schedule):
                break
            if counter % 3 == 0:
                opt = torch.optim.Adam(model.parameters(), lr=lr_schedule[int(counter/3)])
                print("UPDATING LR TO " + str(lr_schedule[int(counter/3)]))
        last = loss.item()   
        
def compute_thetas(model):
    k = sum(list(model.L.values()))
    out = np.zeros((model.n, k))
    
    global_idx = model.L["global"]
    
    tot = global_idx    
    f  = ["global"]*model.L["global"]
    for cell_type in model.cell_types:
        alpha = torch.exp(model.alpha[cell_type]).detach().numpy()
        out[model.labels == cell_type, :global_idx] =  alpha[:,:global_idx]
        out[model.labels == cell_type, tot:tot+model.L[cell_type]] = alpha[:,global_idx:]
        
        tot += model.L[cell_type]

        f = f + [cell_type]*model.L[cell_type]
    return out, f

def return_factor_matrix(model, gs_names = None ,dim = 1):
    k = sum(list(model.L.values()))
    out = np.zeros((k, model.p))
    names = ['na']*k
    
    theta_ct = torch.softmax(model.theta["global"], dim = dim)
    theta = theta_ct.detach().numpy().T
    tot = 0
    out[0:theta.shape[0],:] = theta 
    tot += theta.shape[0]
    if gs_names:
        for i in range(len(gs_names['global'])):
            names[i] = gs_names['global'][i]
    
    for cell_type in model.cell_types:
        theta_ct = torch.softmax(model.theta[cell_type], dim = dim)
        theta = theta_ct.detach().numpy().T
        out[tot:tot+theta.shape[0],:] = theta 
        
        if gs_names:
            for i in range(len(gs_names[cell_type])):
                names[tot + i] = gs_names[cell_type][i]
        
        tot += theta.shape[0]
        
        
    return out, names

def return_markers(factor_matrix, id2word,n_top_vals = 100):
    idx_matrix = np.argsort(factor_matrix,axis = 1)[:,::-1][:,:n_top_vals]
    df3 = pd.DataFrame(np.zeros(idx_matrix.shape))
    for i in range(idx_matrix.shape[0]):
        for j in range(idx_matrix.shape[1]):
            df3.iloc[i,j] = id2word[idx_matrix[i,j]]
    return df3

def jaccard(list1, list2):
    """
    modded
    """
    intersection = len(list(set(list1).intersection(set(list2))))
    union = min(len(list1),len(list2))# + len(list2)) - intersection
    return float(intersection) / union

def matching(markers, gene_names_dict):
    """
    best match based on Jaccard
    """
    matches = []
    jaccards = []
    for i in range(markers.shape[0]):
        max_jacc = 0.0 
        best = ""
        for key in gene_names_dict.keys():
            for gs in gene_names_dict[key].keys():
                t = gene_names_dict[key][gs]
                
                jacc = jaccard(list(markers.iloc[i,:]),t)
                if jacc > max_jacc:
                    max_jacc = jacc
                    best = gs 
        matches.append(best)
        jaccards.append(max_jacc)
    return(matches,jaccards)    
def B_diag(model):
    k = sum(list(model.L.values()))
    out = np.zeros(k)
    
    Bg = model.eta["global"].exp()/(1.0 + model.eta["global"].exp())
    Bg = 0.5*(Bg + Bg.T)
    B = torch.diag(Bg).detach().numpy()
    tot = 0
    out[0:B.shape[0]] = B
    tot += B.shape[0]
    
    for cell_type in model.cell_types:
        Bg = model.eta[cell_type].exp()/(1.0 + model.eta[cell_type].exp())
        Bg = 0.5*(Bg + Bg.T)
        B = torch.diag(Bg).detach().numpy()
        out[tot:tot+B.shape[0]] = B
        
        tot += B.shape[0]
        
        
    return out

def graph_network(model, ct, gs, gene_names_dict,id2word, word2id, thres = 0.65, N = 50):
    net = Network(height='750px', width='100%', bgcolor='#FFFFFF', font_color='black', notebook = True)
    net.barnes_hut()
    eta_global = (model.eta[ct]).exp()/(1.0 + (model.eta[ct]).exp())
    eta_global = 0.5*(eta_global + eta_global.T)
    theta_global = torch.softmax(model.theta[ct], dim = 1)
    mat = contract('il,lj,kj->ik',theta_global,eta_global,theta_global).detach().numpy()
    idxs = []
    for term in gene_names_dict[ct][gs]:
        idxs.append(word2id[term])
    ests = list(set(list(mat[idxs,:].sum(axis = 0).argsort()[::-1][:N]) + idxs))
    ests_names = []
    count = 0 
    for est in ests:
        ests_names.append(id2word[est])
        if est not in idxs:
            net.add_node(count, label = id2word[est], color = '#00ff1e')
        else:
            net.add_node(count, label = id2word[est], color = '#162347')
        count += 1
        
    inferred_mat = mat[ests,:][:,ests]
    for i in range(len(inferred_mat)):
        for j in range(i+1, len(inferred_mat)):
            if inferred_mat[i,j] > thres:
                net.add_edge(i, j)
    neighbor_map = net.get_adj_list()
    for node in net.nodes:
        node['value'] = len(neighbor_map[node['id']])

    return net
