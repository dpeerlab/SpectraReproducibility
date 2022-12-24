#preprocessing functions for single cell data
import scanpy as sc
import pandas as pd
import numpy as np

def calculate_optimal_PC(adata, min_PC = 50, min_var=25, n_comps=100, use_hv=None):
    'select number of PCs based on min_PC and min_var threshold'
    import matplotlib.pyplot as plt
    import scanpy as sc
    from kneed import KneeLocator

    #calculate PCs
    sc.tl.pca(adata, n_comps=100, zero_center=True, svd_solver='arpack', random_state=0, return_info=False, use_highly_variable=use_hv, dtype='float32', copy=False, chunked=False, chunk_size=None)
    sc.pl.pca_variance_ratio(adata, log=False)

    #calculate number of PCs

    a = adata.copy()# PCs kneepoint
    x = [i for i in range(len(a.uns["pca"]["variance_ratio"]))]
    y = list(a.uns["pca"]["variance_ratio"])
    kneedle = KneeLocator(x,
                      y,
                      S=1,
                      curve='convex',
                      direction='decreasing',
                      online=False)
    kn_pc = round(kneedle.knee, 3)
    kneedle.plot_knee()
    plt.show()
    kneedle.plot_knee_normalized()
    plt.show()
    print("Kneepoint happens at PC:", kn_pc)

    exp_var = sum(adata.uns['pca']['variance_ratio'][:kn_pc])
    exp_var_test_percent = exp_var*100
    print(kn_pc,'PC explain',exp_var_test_percent, '% of variance')

    #find number of PCs explaining at least min variance
    tested_PC_number = kn_pc



    tested_PC_number = 1
    while exp_var_test_percent <min_var:
        tested_PC_number = tested_PC_number+1
        exp_var_test = sum(adata.uns['pca']['variance_ratio'][0:tested_PC_number])
        exp_var_test_percent = exp_var_test*100
        print(tested_PC_number, 'PC explain',exp_var_test_percent, '% of variance')
        if tested_PC_number == n_comps:
            break

            
    exp_var_test = sum(adata.uns['pca']['variance_ratio'][0:min_PC])
    exp_var_test_percent = exp_var_test*100
    if tested_PC_number <min_PC:
        print('setting PCs to',min_PC)
        print('variance of',min_PC,'is',exp_var_test_percent,'%')
        tested_PC_number = min_PC
        
        
    #define PC number for embeddings
    number_of_PC_used = tested_PC_number
    
    exp_var_test = sum(adata.uns['pca']['variance_ratio'][0:tested_PC_number])
    exp_var_test_percent = exp_var_test*100
    print('number of PCs for clusterings/embeddings is:', number_of_PC_used)
    print('these explain', exp_var_test_percent,'of variance') 
    
    #recalculate PCs
    sc.pp.pca(adata, n_comps=number_of_PC_used, zero_center=True, svd_solver='arpack', random_state=0, return_info=False, use_highly_variable=use_hv, dtype='float32', copy=False, chunked=False, chunk_size=None)
    sc.pl.pca_variance_ratio(adata, log=False)
    return number_of_PC_used


# Count distribution
def total_molecules_per_cell(adat, count=str):

    # Plot distribution of total molecules per cell
    a = adat.copy()

    print("Median total count of UMI is:", np.median(a.obs[count]))

    # Per cell (Total molecules per cell)
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3,
                                        ncols=1,
                                        sharex=True,
                                        figsize=(15, 5))

    a.obs[count][a.obs[count] != 0].plot(kind="hist",
                                         bins=100,
                                         ax=ax1,
                                         color="r")
    a.obs[count][a.obs[count] != 0].plot(kind="kde", ax=ax2, color="r")
    a.obs[count][a.obs[count] != 0].plot(kind="box",
                                         vert=False,
                                         ax=ax3,
                                         color="r")

    x = ax3.set_xlabel("Molecule counts "+ count +" per cell")

    return None



#Gene correlations/covariance


def corr_cov_matrix_cluster(df,markers=['all_markers'], metric='spearman',clustering_method = 'average', linkage_distance=1.6):
  'calculate a covariance/correlation matrix with flattened hierarchical clustering'
  import scipy
  #define markers to correlate
  if markers == ['all_markers']:
    markers = list(df.index)

  # Filter out genes not present data
  marker_cytokines_filtered = []
  for marker in markers:
    if marker in df.columns:   
      marker_cytokines_filtered.append(marker)

  # Filter out genes not present in data
  a = np.std(df.loc[:,df.columns.isin(marker_cytokines_filtered)], axis=0)
  b = pd.DataFrame(a.T, index=a[df.columns.isin(marker_cytokines_filtered)].index)
  c = b[0]!=0
  d = b[c]
  e = b[0]==0
  f = b[e]
  marker_cytokines_filtered_nonzeroSD = d

  #marker gene list
  marker_cytokines_filtered_nonzeroSD = list(marker_cytokines_filtered_nonzeroSD.index)
    
  #calculate covariance or corrcoef 
  if metric == 'spearman':             
    subsetcluster_cov = np.corrcoef(df.loc[:,marker_cytokines_filtered_nonzeroSD].T)
  elif metric == 'covariance':            
    subsetcluster_cov = np.cov(df.loc[:,marker_cytokines_filtered_nonzeroSD].T)
  else:
    print('please set variable <subsetcluster_cov> to either: spearman, covariance or covariance_log')
    #hierarchically cluster rows and columns    
  subsetcluster_cov_df = pd.DataFrame(subsetcluster_cov, index=marker_cytokines_filtered_nonzeroSD, columns=marker_cytokines_filtered_nonzeroSD)
  if True not in np.isnan(subsetcluster_cov):
      #define hierachical clustering
      row_linkage = scipy.cluster.hierarchy.linkage(scipy.spatial.distance.pdist(subsetcluster_cov), method=clustering_method)
      col_linkage = scipy.cluster.hierarchy.linkage(scipy.spatial.distance.pdist(subsetcluster_cov.T), method=clustering_method)
      #color side bar according to flat clustering
      row_col = []
      col_col = []
  else:
    print('dataframe contains NaNs')
          
  hierarchical_cluster = scipy.cluster.hierarchy.fcluster(row_linkage, linkage_distance, criterion='distance')
  
  print('output list order: dataframe, hierarchical flat clusters, row linkage, column linkage')
  return [subsetcluster_cov_df, hierarchical_cluster, row_linkage, col_linkage]


def plot_cov_clustermap(output,colormap):
    'plot clustermap based on corr_cov_matrix_cluster function output'
    subsetcluster_cov_df = output[0]
    hierarchical_cluster = output[1]
    row_linkage = output[2]
    col_linkage = output[3]
    row_col=[]
    col_col= []
    for j in hierarchical_cluster:
        a = colormap[j]
        row_col.append(a)
        col_col.append(a)

    tmp = sns.clustermap(subsetcluster_cov_df, row_linkage=row_linkage, 
                     col_linkage=col_linkage, col_colors = col_col, row_colors = row_col,
                    figsize=(50,50),vmax=1, center=0,vmin=-1, cmap='seismic')

    return tmp


#aggregate marker genes 

def plt_gene_heatmap_df(adata,pheno_heat,marker_genes,cluster_minus_one_present=False, zscore=True, layer_heat = False):
    
    'create dataframe containing mean zscored gene expression per categorical obs variable'
    import scipy
    #keep only genes in adata
    marker_genes = list(set([x for x in marker_genes if x in adata.var_names]))

    ## calculate means for each cluster and export to dataframe
    all_cells_df_clean = adata.obs[pheno_heat] 
    clusters_cells_clean = list(set(all_cells_df_clean.values))
    
    ## remove -1 cells not belonging to any cluster
    if cluster_minus_one_present == True:
        clusters_cells_clean.remove(-1) #optional if -1 in heatmap
        
    ## create empty dataframe
    marker_df = pd.DataFrame(index=clusters_cells_clean, columns=marker_genes)
    marker_df = marker_df.astype('float')
 
    ## select imputed or raw layer   
    if layer_heat == False:
        if zscore:
            if scipy.sparse.issparse(adata.X):
                gene_expression_zscored = pd.DataFrame(scipy.stats.zscore(np.array(adata.X.todense()), axis=0), index=adata.obs_names, columns=adata.var_names)
            else:
                gene_expression_zscored = pd.DataFrame(scipy.stats.zscore(adata.X, axis=0), index=adata.obs_names, columns=adata.var_names)
        else:
            if scipy.sparse.issparse(adata.X):
                gene_expression_zscored = pd.DataFrame(np.array(adata.X.todense()), index=adata.obs_names, columns=adata.var_names)
            else:
                gene_expression_zscored = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
    else:
        if zscore:
            gene_expression_zscored = pd.DataFrame(scipy.stats.zscore(np.array(adata.layers[layer_heat]),axis=0),
                                                       index=adata.obs_names, columns=adata.var_names) # create zscored gene expression df
        else:
            gene_expression_zscored = pd.DataFrame(np.array(adata.layers[layer_heat]),
                                                       index=adata.obs_names, columns=adata.var_names) 
        
        
    for i in clusters_cells_clean:
        print('Cluster: ',i)
        cluster_adata = all_cells_df_clean[all_cells_df_clean==i]
        cluster_adata2 = gene_expression_zscored.loc[cluster_adata.index, :]
        marker_df.loc[i,:] = np.mean(cluster_adata2, axis=0)
    # drop na columns
    marker_df = marker_df.dropna(axis=1, how='any', thresh=None, subset=None, inplace=False)

    # drop all zero columns
    non_zero = (marker_df != 0).any(axis=0)
    marker_df = marker_df[non_zero.index[non_zero]]#drop zeros columns bc SD will be 0 --> cant divide with 0 for z score
    marker_df = marker_df.reindex(columns=marker_genes)
    return marker_df
