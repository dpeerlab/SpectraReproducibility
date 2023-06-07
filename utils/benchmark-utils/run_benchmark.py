import scanpy as sc
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from opt_einsum import contract
import scipy
import argparse
import os

from simulation_functions import *

def main(args):
    p = args.p
    N = args.n
    n_active_pathways = args.n_pathways
    
    # n_active_pathways = 30
    n_control_pathways = 5
    gene_set_size = 20
    k = 5
    overlap = 0.3
    signal_strength = 15
    lam = 500/((gene_set_size)*(gene_set_size - 1))
    gene_set_FPR = 0
    gene_set_FNR = 0
    model_kwargs = dict(a=0.3, c=0.3)
    n_components = k + n_active_pathways + n_control_pathways
    
    # simulate data
    lam = 500/((gene_set_size)*(gene_set_size - 1))
    data2, A_star2, theta_star2 = simulate_base_data(N= N,k = k,p = p,scale = 25)
    data,base, A_star,gene_sets = create_pathways(n_control_pathways,n_active_pathways=n_active_pathways, gene_set_size=gene_set_size,p =p,N=N,overlap = overlap, signal_strength = signal_strength)
    noisy_gs = noisy_gene_sets(gene_sets, p, gene_set_FNR,gene_set_FPR) 
    X = data + data2


    adj_matrix = np.zeros((p,p))
    for gene_set in noisy_gs:
        for i in gene_set:
            for j in gene_set:
                if i!=j:
                    adj_matrix[i,j] = 1

    I = create_mask(noisy_gs, G_input = len(noisy_gs), D= X.shape[1]).T.astype(int)
    terms = np.array([str(i) for i in range(I.shape[1])])
    
    # run NMF
    if args.method in ["NMF", "nmf"]:
        time, peak_memory = NMF_runtime(X, n_components)
    elif args.method in ["netNMFsc", "netnmfsc", "NETNMFSC"]:
        time, peak_memory = netNMFsc_runtime(X, adj_matrix, p, n_components, n_inits = 1, max_iters = 100000, n_jobs = 1)
    elif args.method in ["scHPF", "schpf", "SCHPF"]:
        time, peak_memory = scHPF_runtime(X, n_components)
    elif args.method in ["expimap_gpu", "expimap", "EXPIMAP"]:
        time, peak_memory = expimap_gpu_runtime(X, I)
    elif args.method in ["expimap_gpu_defaults"]:
        time, peak_memory = expimap_gpu_defaults_runtime(X, I)
    elif args.method in ["spectra_gpu", "spectra", "SPECTRA"]:
        time, peak_memory = spectra_gpu_runtime(X, n_components, p, adj_matrix, noisy_gs)
    elif args.method in ["spectra_cpu"]:
        time, peak_memory = spectra_cpu_runtime(X, n_components, p, adj_matrix, noisy_gs)
    elif args.method in ['slalom', 'SLALOM']:
        vocab = list(range(p))
        time, peak_memory = slalom_runtime(X.astype(float), terms, I, vocab)
    else:
        print("args.method does not exist.")
        
    pd.DataFrame([time, peak_memory], index = ['Time', 'Peak Memory']).T.to_csv(args.out_path)
    return
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m","--method", help="Choose between 'NMF', 'netNMFsc', 'scHPF', 'expimap', 'spectra' ", type=str, default=None)
    parser.add_argument("-p","--p", help = "# of genes (features)", type = int, default = 2000)
    parser.add_argument("-n","--n", help = "# of cells", type = int, default = 300)
    parser.add_argument("--n_pathways", help = "# of active pathways", type = int, default = 30)

    parser.add_argument("-o","--out_path", help = "Path of output file", type = str, default = None)
    args = parser.parse_args()
    main(args)