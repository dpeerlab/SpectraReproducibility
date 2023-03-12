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

def am(genes, word2id):
    n = len(word2id)
    adj_matrix = np.zeros((n,n))
    for gene_set in genes:
        for i in range(len(gene_set)):
            for j in range(len(gene_set)):
                g1 = gene_set[i]
                g2 = gene_set[j]
                adj_matrix[word2id[g1],word2id[g2]] = 1
    return adj_matrix

def am_weighted(genes, word2id):
    n = len(word2id)
    adj_matrix = np.zeros((n,n))
    ws = []
    for gene_set in genes:
        if len(gene_set) > 1:
            w = 1.0/(len(gene_set)*(len(gene_set)-1)/2.0)
        else:
            w = 1.0
        ws.append(w)
        for i in range(len(gene_set)):
            for j in range(len(gene_set)):
                g1 = gene_set[i]
                g2 = gene_set[j]
                adj_matrix[word2id[g1],word2id[g2]] += w
    med = np.median(np.array(ws))
    return adj_matrix/float(med)

def corr_single(w1, w2, W):
    eps = 0.01
    dw1 = W[:, w1]
    dw2 = W[:, w2]
   
    correlation = np.corrcoef(dw1[(dw1 > 0)&(dw2 > 0)], dw2[(dw1 > 0)&(dw2 > 0)])[0,1]
    if np.isnan(correlation):
        correlation = 0.0
    return correlation


# calc coherence of topics based on W
# See appendix of https://arxiv.org/pdf/1910.05495.pdf for details
def corr(topics, W):
    score = 0
    count = 0
    K, V = topics.shape[0], topics.shape[1]
    for i in range(K):
        topic = topics[i]
        for j1 in range(len(topic) - 1):
            for j2 in range(j1 + 1, len(topic)):
                sc = corr_single(topic[j1], topic[j2], W)
                score += sc
    return score / ((K * V * (V - 1) / 2))

def corr_median(topics, W):
    score = 0
    count = 0
    K, V = topics.shape[0], topics.shape[1]
    for i in range(K):
        topic = topics[i]
        gg = []
        for j1 in range(len(topic) - 1):
            for j2 in range(j1 + 1, len(topic)):
                sc = corr_single(topic[j1], topic[j2], W)
                gg.append(sc)
        score += np.median(np.array(gg))
    return score / K

def corr_mean(topics, W):
    score = 0
    count = 0
    K, V = topics.shape[0], topics.shape[1]
    for i in range(K):
        topic = topics[i]
        gg = []
        for j1 in range(len(topic) - 1):
            for j2 in range(j1 + 1, len(topic)):
                sc = corr_single(topic[j1], topic[j2], W)
                gg.append(sc)
        score += np.mean(np.array(gg))
    return score / K

def coh_median(topics, W):
    score = 0
    count = 0
    K, V = topics.shape[0], topics.shape[1]
    for i in range(K):
        topic = topics[i]
        gg = []
        for j1 in range(len(topic) - 1):
            for j2 in range(j1 + 1, len(topic)):
                sc = coherence_single(topic[j1], topic[j2], W)
                gg.append(sc)
        score += np.median(np.array(gg))
    return score / K

def corr_ind(topics, W):
    scores = []
    count = 0
    K, V = topics.shape[0], topics.shape[1]
    for i in range(K):
        score = 0
        topic = topics[i]
        for j1 in range(len(topic) - 1):
            for j2 in range(j1 + 1, len(topic)):
                sc = corr_single(topic[j1], topic[j2], W)
                score += sc
        scores.append(score)
    return np.array(scores) / ((V * (V - 1) / 2))

def coherence_single(w1, w2, W):
    if (w1 == -1)|(w2 == -1):
        return -1
    eps = 0.01
    dw1 = W[:, w1] > 0
    dw2 = W[:, w2] > 0
    N = W.shape[0]

    dw1w2 = (dw1 & dw2).float().sum() / N + eps
    dw1 = dw1.float().sum() / N + eps
    dw2 = dw2.float().sum() / N + eps

    return dw1w2.log() - dw1.log() - dw2.log()


# calc coherence of topics based on W
# See appendix of https://arxiv.org/pdf/1910.05495.pdf for details
def coherence(topics, W):
    score = 0
    count = 0
    tot = 0
    K, V = topics.shape[0], topics.shape[1]
    for i in range(K):
        topic = topics[i]
        for j1 in range(len(topic) - 1):
            for j2 in range(j1 + 1, len(topic)):
                sc = coherence_single(topic[j1], topic[j2], W)
                if sc == -1:
                    tot += 1
                else:
                    score += sc
    return score / ((K * V * (V - 1) / 2) - tot)




def markers_out(output, id2word,n_top_markers):
    sorted_args = np.argsort(output)[:,::-1][:,:n_top_markers]
    ans = pd.DataFrame(np.zeros((sorted_args.shape[0], sorted_args.shape[1])))
    for i in range(sorted_args.shape[0]):
        for j in range(sorted_args.shape[1]):
            ans.iloc[i,j] = id2word[sorted_args[i,j]]
    return ans 
def markers_args(output, n_top_markers):
    sorted_args = np.argsort(output)[:,::-1][:,:n_top_markers]
    return sorted_args
       


def coherence_ind(topics, W):
    scores = []
    count = 0
    tot = 0
    K, V = topics.shape[0], topics.shape[1]
    for i in range(K):
        score = 0
        topic = topics[i]
        for j1 in range(len(topic) - 1):
            for j2 in range(j1 + 1, len(topic)):
                sc = coherence_single(topic[j1], topic[j2], W)
                if sc == -1:
                    tot += 1
                else:
                    score += sc
        scores.append(score)
    return np.array(scores) / ((V * (V - 1) / 2) - tot)


def find_neighborhoods(adata, n_index = 200, n_neighbors = 200):
    sc.pp.neighbors(adata, n_neighbors = n_neighbors)
    amatrix = np.array((adata.obsp["connectivities"] >0).todense())
    index_set = np.random.permutation(amatrix.shape[0])[:n_index] 
    
    masks = amatrix[index_set,:]
    idxs = [] 
    for i in range(masks.shape[0]):
        mask1 = masks[i,:]
        mn = np.array(adata[mask1,:].X.mean(axis = 0))
        dists = ((adata.X - mn.reshape(1,-1))**2).sum(axis = 1)
        idxs.append(np.argmin(dists))
    idxs = np.unique(np.array(idxs))
    final_masks = amatrix[idxs,:]
    return final_masks

def max_corr(neighborhoods, adata, g1, g2):
    max_coef = 0.0
    for i in range(neighborhoods.shape[0]):
        temp_adata = adata[neighborhoods[i],:]
        vec1 = np.array(temp_adata[:,g1].X).ravel()
        vec2 = np.array(temp_adata[:,g2].X).ravel()
        cc = np.corrcoef(vec1,vec2)[0,1]
        if cc > max_coef:
            max_coef = cc
    return max_coef

def max_corr_single(neighborhoods, w1, w2, W):
    max_coef = 0.0
    dw1 = W[:, w1]
    dw2 = W[:, w2]
    
    for i in range(neighborhoods.shape[0]):
        temp_w1 = W[neighborhoods[i],w1]
        temp_w2 = W[neighborhoods[i],w2]
        cc = np.corrcoef(temp_w1[(temp_w1 > 0)&(temp_w2 > 0)],temp_w2[(temp_w1 > 0)&(temp_w2 > 0)])[0,1]
        if cc > max_coef:
            max_coef = cc
    return max_coef 
    
def max_corr_median(neighborhoods,topics, W):
    score = 0
    count = 0
    K, V = topics.shape[0], topics.shape[1]
    for i in range(K):
        topic = topics[i]
        gg = []
        for j1 in range(len(topic) - 1):
            for j2 in range(j1 + 1, len(topic)):
                sc = max_corr_single(neighborhoods,topic[j1], topic[j2], W)
                gg.append(sc)
        score += np.median(np.array(gg))
    return score / K

def process_Bassez(hv = 3000,pseudocount = ):#3.0):
    #FILE = "/data/TRAIN_BRCA-X-TIL-X-Bassez_2021-X-cohort1_2_raw_filtered_clustered_drops_annotated_nodrops_log1p_clustered_leukocytes_scran_annotated_clustered_imputed_hvgenes_andmarker_15000_clustered_imputed_v2_210501_annotated_211208.h5ad"
    FILE = "BRCA-X-TIL-X-Bassez_2021-X-cohort1_2_raw_filtered_clustered_drops_annotated_nodrops_log1p_clustered_leukocytes_scran_annotated_clustered_imputed_hvgenes_andmarker_15000_clustered_imputed_v2_210501_annotated_211208.h5ad"
    adata = sc.read_h5ad(FILE)
    f1 = pd.read_csv("data/SPADE_cells_x_genesets_celltype-specific_modified_for_SPADE.csv")
    f2 = pd.read_csv("data/SPADE_cells_x_genesets_general.csv")
    f3 = pd.read_csv("data/SPADE_genes_x_genesets.csv")
    

    adata.var['ighm'] = adata.var_names.str.startswith('IGHM')
    adata.var['iglc'] = adata.var_names.str.startswith('IGLC')
    adata.var['ighg'] = adata.var_names.str.startswith('IGHG')
    adata.var['igha'] = adata.var_names.str.startswith('IGHA')



    adata.var['ighv'] = adata.var_names.str.startswith('IGHV')
    adata.var['iglv'] = adata.var_names.str.startswith('IGLV')
    adata.var['igkv'] = adata.var_names.str.startswith('IGKV')
    adata.var['trbv'] = adata.var_names.str.startswith('TRBV')
    adata.var['trav'] = adata.var_names.str.startswith('TRAV')
    adata.var['trgv'] = adata.var_names.str.startswith('TRGV')
    adata.var['trdv'] = adata.var_names.str.startswith('TRDV')

    adata.var['hb'] = adata.var_names.str.startswith('HB')

    adata = adata[:,~(adata.var['hb']|adata.var['ighm']|adata.var['iglc']|adata.var['ighg']|adata.var['igha']|adata.var['ighv']|adata.var['trgv']|adata.var['trdv']|adata.var['iglv']|adata.var['igkv']|adata.var['trav']|adata.var['trbv'])]
    sc.pp.highly_variable_genes(adata, n_top_genes = hv, flavor = "cell_ranger")
    
    #remove genes that don't appear in the data
    gene_lst = adata.var.index
    remove = []
    genes = []
    for gene_name in f3["g.name"]:
        if gene_name not in gene_lst:
            f3 = f3[f3["g.name"] != gene_name]

    full_genes = list(f3["g.name"])
    bools = []
    for name in adata.var.index:
        if name in full_genes:
            bools.append(True)
        else:
            bools.append(False)
    bools = np.array(bools)
    adata2 = adata[:,adata.var.highly_variable.values|bools]
    #do we exponentiate X?
    #X = np.exp(np.array(adata2.X))
    X = np.array(adata2.X)
    X = X + pseudocount
    vocab = adata2.var_names
    word2id = dict((v, idx) for idx, v in enumerate(vocab))
    id2word = dict((idx, v) for idx, v in enumerate(vocab))
    labels = np.array(adata2.obs.annotation_SPADE_1)

    adict = OrderedDict()
    weights = OrderedDict()
    gs_dict = OrderedDict()
    gene_names_dict = OrderedDict()
    gs_names = OrderedDict()


    lst = []
    names = []
    for gs in f2["gs.name"].unique():
        gene_set = list(set(f3[f3["gs.name"] == gs]["g.name"]))
        if len(gene_set) > 0:
            lst.append(gene_set)
            names.append(gs)
    adict["global"] = am(lst,word2id)
    weights["global"] = am_weighted(lst,word2id)
    gs_dict["global"] = [[word2id[i] for i in j] for j in lst]
    gene_names_dict["global"] = {names[i]:lst[i] for i in range(len(lst))}
    gs_names["global"] = names

    for ct in np.unique(labels):
        f = f1[f1["c.name"] == ct]
        if len(f) > 0:
            lst = []
            names = []
            for gs in f["gs.name"].unique():
                gene_set = list(set(f3[f3["gs.name"] == gs]["g.name"]))
                if len(gene_set) > 0:
                    lst.append(gene_set)
                    names.append(gs)
            adict[ct] = am(lst, word2id)
            weights[ct] = am_weighted(lst,word2id)
            gs_dict[ct] = [[word2id[i] for i in j] for j in lst]
            gene_names_dict[ct] = {names[i]:lst[i] for i in range(len(lst))}
            gs_names[ct] = names
        else:
            adict[ct] = []
            weights[ct] = [] 
            gs_dict[ct] = []
            gs_names[ct] = []
    return X,adata2, word2id, id2word, labels, vocab, adict, weights,gene_names_dict, gs_dict, gs_names


def process_Zhang(hv = 3000):#,pseudocount = 3.0):
    FILE = "data/TIL-X-BRCA-X-Zhang-X-2021-X-all_cells_raw_annotated_clustered_nodrops_nodoub_tumor_scran_labeled_clustered_annotated_211208_removed-doublets.h5ad"
    adata = sc.read_h5ad(FILE)
    f1 = pd.read_csv("data/SPADE_cells_x_genesets_celltype-specific_modified_for_SPADE.csv")
    f2 = pd.read_csv("data/SPADE_cells_x_genesets_general.csv")
    f3 = pd.read_csv("data/SPADE_genes_x_genesets.csv")
    

    adata.var['ighm'] = adata.var_names.str.startswith('IGHM')
    adata.var['iglc'] = adata.var_names.str.startswith('IGLC')
    adata.var['ighg'] = adata.var_names.str.startswith('IGHG')
    adata.var['igha'] = adata.var_names.str.startswith('IGHA')



    adata.var['ighv'] = adata.var_names.str.startswith('IGHV')
    adata.var['iglv'] = adata.var_names.str.startswith('IGLV')
    adata.var['igkv'] = adata.var_names.str.startswith('IGKV')
    adata.var['trbv'] = adata.var_names.str.startswith('TRBV')
    adata.var['trav'] = adata.var_names.str.startswith('TRAV')
    adata.var['trgv'] = adata.var_names.str.startswith('TRGV')
    adata.var['trdv'] = adata.var_names.str.startswith('TRDV')

    adata.var['hb'] = adata.var_names.str.startswith('HB')

    adata = adata[:,~(adata.var['hb']|adata.var['ighm']|adata.var['iglc']|adata.var['ighg']|adata.var['igha']|adata.var['ighv']|adata.var['trgv']|adata.var['trdv']|adata.var['iglv']|adata.var['igkv']|adata.var['trav']|adata.var['trbv'])]
    sc.pp.highly_variable_genes(adata, n_top_genes = hv, flavor = "cell_ranger")
    
    #remove genes that don't appear in the data
    gene_lst = adata.var.index
    remove = []
    genes = []
    for gene_name in f3["g.name"]:
        if gene_name not in gene_lst:
            f3 = f3[f3["g.name"] != gene_name]

    full_genes = list(f3["g.name"])
    bools = []
    for name in adata.var.index:
        if name in full_genes:
            bools.append(True)
        else:
            bools.append(False)
    bools = np.array(bools)
    adata2 = adata[:,adata.var.highly_variable.values|bools]
    #do we exponentiate X?
    #X = np.exp(np.array(adata2.X))
    X = np.array(adata2.X.todense())
    X = X + pseudocount
    vocab = adata2.var_names
    word2id = dict((v, idx) for idx, v in enumerate(vocab))
    id2word = dict((idx, v) for idx, v in enumerate(vocab))
    labels = np.array(adata2.obs.annotation_SPADE_1)

    adict = OrderedDict()
    weights = OrderedDict()
    gs_dict = OrderedDict()
    gene_names_dict = OrderedDict()
    gs_names = OrderedDict()


    lst = []
    names = []
    for gs in f2["gs.name"].unique():
        gene_set = list(set(f3[f3["gs.name"] == gs]["g.name"]))
        if len(gene_set) > 0:
            lst.append(gene_set)
            names.append(gs)
    adict["global"] = am(lst,word2id)
    weights["global"] = am_weighted(lst,word2id)
    gs_dict["global"] = [[word2id[i] for i in j] for j in lst]
    gene_names_dict["global"] = {names[i]:lst[i] for i in range(len(lst))}
    gs_names["global"] = names

    for ct in np.unique(labels):
        f = f1[f1["c.name"] == ct]
        if len(f) > 0:
            lst = []
            names = []
            for gs in f["gs.name"].unique():
                gene_set = list(set(f3[f3["gs.name"] == gs]["g.name"]))
                if len(gene_set) > 0:
                    lst.append(gene_set)
                    names.append(gs)
            adict[ct] = am(lst, word2id)
            weights[ct] = am_weighted(lst,word2id)
            gs_dict[ct] = [[word2id[i] for i in j] for j in lst]
            gene_names_dict[ct] = {names[i]:lst[i] for i in range(len(lst))}
            gs_names[ct] = names
        else:
            adict[ct] = []
            weights[ct] = [] 
            gs_dict[ct] = []
            gs_names[ct] = []
    return X,adata2, word2id, id2word, labels, vocab, adict, weights,gene_names_dict, gs_dict, gs_names
