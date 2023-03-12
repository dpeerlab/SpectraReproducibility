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

#spade imports
from pyspade_global import *
from util import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lam', type=float, required=True)
parser.add_argument('--delta', type=float, required=True)
parser.add_argument('--pseudocount',type =float,required=True)
parser.add_argument('--beta',type = float,required=True)
args = parser.parse_args()
lam = args.lam
delta = args.delta
pseudocount = args.pseudocount
beta = args.beta

print(lam)
print(delta)
print(beta)
X, adata2, word2id, id2word, labels, vocab, adict, weights, gene_names_dict, gs_dict, gs_names = process_Bassez(pseudocount = pseudocount)


L = OrderedDict({"global": len(gs_names["global"]) + 2})
for k in np.unique(labels):
    L[k] = len(gs_names[k]) + 1
    if k == 'nan':
        L[k] = 0

model = SPADE(X = X,L = L,labels = labels,adj_matrix = adict, weights = weights,lam = lam,kappa = 0.00001,rho = 0.001,delta = delta,beta=beta)
model.initialize_total(gs_dict, 25)
train(model)
torch.save(model.state_dict(), "global" + str(lam) + "_delta_"+ str(delta)+ "_beta_" + str(beta) + "_pseudo_" +str(pseudocount))
#https://pytorch.org/tutorials/beginner/saving_loading_models.html

