import scanpy as sc
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from opt_einsum import contract
import time
import scipy
import argparse
import os
import tracemalloc
import collections
import anndata
from spectra import spectra_gpu_mod as spc
from util import *


def main(args):
    num_cell_types = args.num_cell_types
    num_cells = 25000
    num_genes = 3000
    num_global_factors = 10
    num_cell_type_specific_factors = int(128/num_cell_types)
    
    cache_dir = args.cache_dir
    
    with open(f'{cache_dir}{num_cell_type_specific_factors}_ctf_{num_cell_types}_ct_annotations.json', 'r') as read_file:
        annotations = json.loads(read_file.read())
    adata = sc.read_h5ad(f'{cache_dir}{num_cell_type_specific_factors}_ctf_{num_cell_types}_ct.h5ad')
    tracemalloc.start()
    start1 = time.time()
    
    model = spc.est_spectra(adata = adata, gene_set_dictionary = annotations, 
                        use_highly_variable = False, cell_type_key = "cell_type", 
                        use_weights = True, lam = 0.01, 
                        delta=0.001,kappa = 0.00001, rho = 0.00001, 
                        use_cell_types = True, n_top_vals = 25, 
                        num_epochs=10000 #for demonstration purposes we will only run 2 epochs, we recommend 10,000 epochs
                       )
    
    end = time.time()
    peak_memory = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    time_passed = end - start1
    pd.DataFrame([time_passed, peak_memory], index = ['Time', 'Peak Memory']).T.to_csv(args.out_path)    
    return
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_cell_types", help = "# of celltypes", type = int, default = None)
    parser.add_argument("--cache_dir", help="cache_dir", type = str, default = None)
    # parser.add_argument("--target", help = "# of cells", type = int, default = 50000)

    parser.add_argument("-o","--out_path", help = "Path of output file", type = str, default = None)
    args = parser.parse_args()
    main(args)