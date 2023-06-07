import scanpy as sc
import pandas as pd
import numpy as np
import seaborn as sns
# import matplotlib.pyplot as plt
from opt_einsum import contract
import time
import scipy
from sklearn.decomposition import NMF

import tracemalloc

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

# --------------- SIMULATIONS ------------------------

def NMF_runtime(X, n_components):
    tracemalloc.start()
    start = time.time()
    nmf = NMF(n_components = n_components)
    nmf.fit(X.astype(float))
    end = time.time()
    peak_memory = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return end - start, peak_memory

def netNMFsc_runtime(X, adj_matrix, p, dimensions, n_inits = 1, max_iters = 100000, n_jobs = 1):
    import netNMFsc
    tracemalloc.start()
    start = time.time()
    operator = netNMFsc.netNMFGD(d=dimensions, n_inits=n_inits, max_iter=max_iters, n_jobs=n_jobs)
    operator.N = adj_matrix
    operator.X = X.T
    operator.genes = [str(i) for i in range(p)]
    W = operator.fit_transform()
    end = time.time()
    peak_memory = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return end - start, peak_memory

def scHPF_runtime(X, n_components):
    import schpf
    tracemalloc.start()
    start = time.time()
    fib_coo = scipy.sparse.coo_matrix(X)
    fib_hpf = schpf.scHPF(nfactors=n_components, verbose=True)
    fib_hpf.fit(fib_coo)
    end = time.time()
    peak_memory = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return end - start, peak_memory

def expimap_gpu_defaults_runtime(X, I):
    import scarches as sca
    tracemalloc.start()
    start = time.time()
    
    adata = sc.AnnData(X)
    adata.varm['I'] = I
    adata.obs['study'] = 'same'
    adata._inplace_subset_var(adata.varm['I'].sum(1)>0)

    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    
    if adata.shape[1] > 2000:
        n_top_genes = 2000
    else:
        n_top_genes = adata.shape[1]
        
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=2000,
        batch_key="study",
        subset=True)
    
    adata._inplace_subset_var(adata.varm['I'].sum(1)>0)
    
    start = time.time()
    intr_cvae = sca.models.EXPIMAP(
        adata=adata,
        condition_key='study',
        hidden_layer_sizes=[256, 256, 256],
        recon_loss='nb',
        #soft_mask =True ########this is important to modify the gene sets
    )

    ALPHA = 0.7

    early_stopping_kwargs = {
        "early_stopping_metric": "val_unweighted_loss", # val_unweighted_loss
        "threshold": 0,
        "patience": 50,
        "reduce_lr": True,
        "lr_patience": 13,
        "lr_factor": 0.1,
    }
    intr_cvae.train(
        n_epochs=400,
        alpha_epoch_anneal=100,
        alpha=ALPHA,
        alpha_kl=0.5,
        weight_decay=0.,
        # alpha_l1=0.4, ########this is important to modify the gene sets
        early_stopping_kwargs=early_stopping_kwargs,
        use_early_stopping=True,
        monitor_only_val=False,
        seed=2020,
    )
    end = time.time()
    peak_memory = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return end - start, peak_memory

def expimap_gpu_runtime(X, I):
    import scarches as sca
    tracemalloc.start()
    start = time.time()
    
    adata = sc.AnnData(X)
    adata.varm['I'] = I
    adata.obs['study'] = 'same'
    adata._inplace_subset_var(adata.varm['I'].sum(1)>0)

    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    
    if adata.shape[1] > 2000:
        n_top_genes = 2000
    else:
        n_top_genes = adata.shape[1]
        
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=2000,
        batch_key="study",
        subset=True)
    
    adata._inplace_subset_var(adata.varm['I'].sum(1)>0)
    
    start = time.time()
    intr_cvae = sca.models.EXPIMAP(
        adata=adata,
        condition_key='study',
        hidden_layer_sizes=[256, 256, 256],
        recon_loss='nb',
        soft_mask =True ########this is important to modify the gene sets
    )

    ALPHA = 0.7

    early_stopping_kwargs = {
        "early_stopping_metric": "val_unweighted_loss", # val_unweighted_loss
        "threshold": 0,
        "patience": 50,
        "reduce_lr": True,
        "lr_patience": 13,
        "lr_factor": 0.1,
    }
    intr_cvae.train(
        n_epochs=400,
        alpha_epoch_anneal=100,
        alpha=ALPHA,
        alpha_kl=0.5,
        weight_decay=0.,
        alpha_l1=0.4, ########this is important to modify the gene sets
        early_stopping_kwargs=early_stopping_kwargs,
        use_early_stopping=True,
        monitor_only_val=False,
        seed=2020,
    )
    end = time.time()
    peak_memory = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return end - start, peak_memory

def spectra_gpu_runtime(X, n_components, p, adj_matrix, noisy_gs):
    from spectra import spectra_gpu_mod as spc
    import torch
    tracemalloc.start()
    start1 = time.time()
    model = spc.SPECTRA_Model(X, labels = None, L = n_components, adj_matrix = adj_matrix,
                              lam = 0.01, delta=0.001,kappa = 0.00001, rho = 0.00001,
                              use_cell_types = False, device = 'cuda')
    print(f"model create: {time.time() - start1}")
    word2id = dict(zip(range(p), range(p)))
    gs ={}
    count = 0
    for l in noisy_gs:
        gs[f"gene_set_{count}"] = l
        count +=1 
    
    
    gene_set_dictionary = gs
    init_scores = None
    start2 = time.time()
    model.initialize(gene_set_dictionary, word2id, X, init_scores)
    print(f"model initialize: {time.time() - start2}")
    start3 = time.time()
    model.internal_model.to('cuda')
    X = torch.Tensor(X).to('cuda')
    print(f"model load to cuda: {time.time() - start3}")

    start = time.time()
    model.train(X = X, labels = None)
    end = time.time()
    peak_memory = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return end - start, peak_memory

def spectra_cpu_runtime(X, n_components, p, adj_matrix, noisy_gs):
    from spectra import spectra_gpu_mod as spc
    tracemalloc.start()
    model = spc.SPECTRA_Model(X, labels = None, L = n_components, adj_matrix = adj_matrix,
                              lam = 0.01, delta=0.001,kappa = 0.00001, rho = 0.00001,
                              use_cell_types = False, device = 'cpu')
    word2id = dict(zip(range(p), range(p)))
    gs ={}
    count = 0
    for l in noisy_gs:
        gs[f"gene_set_{count}"] = l
        count +=1 
    
    
    gene_set_dictionary = gs
    init_scores = None
    model.initialize(gene_set_dictionary, word2id, X, init_scores)
    start3 = time.time()

    start = time.time()
    model.train(X = X, labels = None)
    end = time.time()
    peak_memory = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return end - start, peak_memory
    
def slalom_runtime(X, terms, I, vocab):
    import slalom
    tracemalloc.start()
    start = time.time()
    
    FA = slalom.initFA(Y = X.astype(float), terms = terms, I = I, gene_ids=vocab, noise='gauss', 
            nHidden=0, nHiddenSparse=0,do_preTrain=False, minGenes = 1, pruneGenes = False)
    FA.train()
    FA.printDiagnostics()
    
    end = time.time()
    peak_memory = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return end - start, peak_memory