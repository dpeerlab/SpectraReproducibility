import scanpy as sc
import pandas as pd
import numpy as np

########################### general #################################################################################
def randomize_cells(anndata_object):
    "Randomize cells for plotting "
    index_list = np.arange(anndata_object.shape[0])
    np.random.shuffle(index_list)
    anndata_object = anndata_object[index_list]
    return anndata_object

def overlap_coefficient(set_a,set_b):
    min_len = min([len(set_a),len(set_b)])
    intersect_len = len(set_a.intersection(set_b))
    overlap = intersect_len/min_len
    return overlap

########################### plot factor cell scores #############################################################

def draw_square_gate(df, plot, group_names, x_name, y_name, xminmax, yminmax, width=2, color='#000000',print_absolute=False):
    '''draw a gate in seaborn jointplot
    df: input dataframe for seaborn.jointplot
    group_names: name of group column in dataframe df to calculate percentages 
    x_name: name of x axis column in dataframe df
    y_name: name of y axis columns in dataframe df
    plot: jointplot object
    xminmax: tuple (start,end) of gate x axis
    yminmax: tuple (start,end) of gate y axis
    color: color of line
    print_absolute: print counts instead of %'''
    #plot.ax_joint.axline((xminmax[0], xminmax[0]),(xminmax[0], yminmax[1]),linewidth=width, color=color)
    #plot.ax_joint.axline((xminmax[0], yminmax[1]),(xminmax[1], yminmax[1]),linewidth=width, color=color)
    #plot.ax_joint.axline((xminmax[1], yminmax[1]),(xminmax[1], yminmax[0]),linewidth=width, color=color)
    #plot.ax_joint.axline((xminmax[1], yminmax[0]),(xminmax[0], yminmax[0]),linewidth=width, color=color)
    
    
    #calculate cells in gat
    groups = list(set(df[group_names]))
    n_allcells = len(df[group_names])
    n_groups = []
    group_name_list = []

    ## calculate cells in gat for all cells

    n_ingate = df[df[x_name]>xminmax[0]]
    n_ingate = n_ingate[n_ingate[x_name]<xminmax[1]]
    n_ingate = n_ingate[n_ingate[y_name]>yminmax[0]]
    n_ingate = n_ingate[n_ingate[y_name]<yminmax[1]]
    group_name_list.append('all')
    n_groups.append((n_allcells,len(n_ingate)))


    for i in groups:
        group_name_list.append(i)
        group_df = df[df[group_names]==i]
        n_group = len(group_df)
        n_ingate = group_df[group_df[x_name]>xminmax[0]]
        n_ingate = n_ingate[n_ingate[x_name]<xminmax[1]]
        n_ingate = n_ingate[n_ingate[y_name]>yminmax[0]]
        n_ingate = n_ingate[n_ingate[y_name]<yminmax[1]]
        n_ingate = len(n_ingate)
        n_groups.append((n_group,n_ingate))
    
    #add gate
    plot.ax_joint.plot((xminmax[0], xminmax[0]),(yminmax[0], yminmax[1]),linewidth=width, color=color)
    plot.ax_joint.plot((xminmax[1], xminmax[1]),(yminmax[0], yminmax[1]),linewidth=width, color=color)
    plot.ax_joint.plot((xminmax[0], xminmax[1]),(yminmax[1], yminmax[1]),linewidth=width, color=color)
    plot.ax_joint.plot((xminmax[0], xminmax[1]),(yminmax[0], yminmax[0]),linewidth=width, color=color)
    
    title = []
    for i,v in enumerate(group_name_list):
        title.append(v+'_:_'+(str((n_groups[i][1]/n_groups[i][0])*100))+'%')
    print(title)
    
    return plot

########################### aggregate factor cell scores ########################################################

def aggregate_cell_scores(adata,clinical_var_obs,clinical_var2_obs,factor_name_list_corr,batch_key,patient_obs,
                            obs_columns,
                            zero_cutoff =0.001):
    '''
    aggregate factor cell scores per batch/patient:
    clinical_var_obs: for later fold change calculation --> #obs for clinical variable to calculate fold change over
    clinical_var2_obs #additional clinical variable
    patient_obs #key for patient id in adata.obs 
    batch_key #key for batch / sample in adata.obs
    factor_name_list_corr #list with factors to calculate fold change for
    zero_cutoff #mean of positive fraction per factor will be calculated, define threshold for positive frac if None mean without any cutoff will be calculated 
     obs_columns: name for columns in output dataframe (order will be clinical_var_obs, clinical_var2_obs, partient_obs)
    '''
    #first calculate mean of positive fraction for each sample
    df_batches = pd.DataFrame()

    for j in set(adata.obs[batch_key]):
        adata_subset = adata[adata.obs[batch_key]==j]
        clinical_var = list(set(adata_subset.obs[clinical_var_obs]))[0]
        clinical_var2 = list(set(adata_subset.obs[clinical_var2_obs]))[0]
        patient = list(set(adata_subset.obs[patient_obs]))[0]

        #define here which metric you want to use
        for i in factor_name_list_corr:
            #greater_zero = list(set(adata_CD8_subset.obs[i] > zero_cutoff))
            #if greater_zero[0] == False and len(greater_zero) == 1:
             #   a = 0 #set 0 if all loadings are 0
            #else:
            #a = scipy.stats.mstats.gmean(adata_CD8_pre.obs[i])
            #a = len(adata_CD8_pre.obs[adata_CD8_pre.obs[i] >zero_cutoff])/len(adata_CD8_pre.obs[i])
            #a = scipy.stats.mstats.gmean(adata_CD8_subset.obs[adata_CD8_subset.obs[i] >zero_cutoff][i])
            if zero_cutoff == None:
                a = np.mean(adata_subset.obs[i])
            else:
                a = np.mean(adata_subset.obs[adata_subset.obs[i] >zero_cutoff][i])
            #a = np.mean(adata_CD8_pre.obs[i])

            df_batches.loc[j,i]=a
            df_batches.loc[j,obs_columns[0]] = clinical_var
            df_batches.loc[j,obs_columns[1]] = clinical_var2
            df_batches.loc[j,obs_columns[2]] = patient
    

    return df_batches




################ plot factor cell score trends along a continuous variable ######################

from collections.abc import Iterable
from scipy.sparse import issparse
from joblib import Parallel, delayed
import rpy2

def _gam_fit_predict(x, y, weights=None, pred_x=None):

    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri, Formula
    from rpy2.robjects.packages import importr

    pandas2ri.activate()

    # Weights
    if weights is None:
        weights = np.repeat(1.0, len(x))

    # Construct dataframe
    use_inds = np.where(weights > 0)[0]
    r_df = pandas2ri.py2rpy(
        pd.DataFrame(np.array([x, y]).T[use_inds, :], columns=["x", "y"])
    )

    # Fit the model
    rgam = importr("gam")
    model = rgam.gam(Formula("y~s(x)"), data=r_df, weights=pd.Series(weights[use_inds]))

    # Predictions
    if pred_x is None:
        pred_x = x
    y_pred = np.array(
        robjects.r.predict(
            model, newdata=pandas2ri.py2rpy(pd.DataFrame(pred_x, columns=["x"]))
        )
    )

    # Standard deviations
    p = np.array(
        robjects.r.predict(
            model, newdata=pandas2ri.py2rpy(pd.DataFrame(x[use_inds], columns=["x"]))
        )
    )
    n = len(use_inds)
    sigma = np.sqrt(((y[use_inds] - p) ** 2).sum() / (n - 2))
    stds = (
        np.sqrt(1 + 1 / n + (pred_x - np.mean(x)) ** 2 / ((x - np.mean(x)) ** 2).sum())
        * sigma
        / 2
    )

    return y_pred, stds


def compute_gene_trends(
    adata:sc.AnnData,
    features:Iterable,
    trajectory:str, 
    n_jobs:int=-1,
    imputed_layer:str=None,):
    """Function for computing gene expression trends along trajectory
    :param adata: AnnData object
    :param features: obs or genes for which to compute expression trends; index in var
    :param trajectory: column in obs which orders cells along trajectory  
    :param layer: layer of AnnData containing imputed gene expression values
    :return: Dictionary of gene expression trends and standard deviations for each branch
    """
    
    # Bin cells along trajectory
    bins = np.linspace(adata.obs[trajectory].min(), adata.obs[trajectory].max(), 250)
    print(features,bins)
    trends = pd.DataFrame(
        0.0, index=features, columns=bins,
    )
    print(trends)
    std = pd.DataFrame(
        0.0, index=features, columns=bins,
    )
    
    # Cast imputed expression to dense matrix
    genes = adata.var.index.intersection(features) # genes
    imp_exprs = adata[:, genes].X if not imputed_layer else adata[:, genes].layers[imputed_layer]
    if issparse(imp_exprs): imp_exprs = imp_exprs.todense()
    
    # Get observations from AnnData
    obs = adata.obs.columns.intersection(features) # observations
    obs_vals = adata.obs[obs].values
    
    # Combine into feature matrix
    feature_mtx = pd.DataFrame(
        np.concatenate([imp_exprs.T, obs_vals.T]),
        index = list(genes) + list(obs),
    ).loc[features]
    print(feature_mtx)
    print(n_jobs)
    print(trajectory)
    #start = time.time()
    
    # Branch cells and weights
    res = Parallel(n_jobs=n_jobs)(
        delayed(_gam_fit_predict)(
            adata.obs[trajectory].values,
            feature_mtx.loc[feature],
            pred_x=bins,
        )
 
        for feature in features
    )
    
    # Fill in the matrices
    print(res)
    for i, feature in enumerate(features):
        trends.loc[feature, :] = res[i][0]
        std.loc[feature, :] = res[i][1]
    
    #end = time.time()
    #print(f"Time for processing: {(end - start) / 60} minutes")
          
    return trends, std


'''
    def compute_factor_trends(adata_trends, factor_of_interest, trend):

        compute factor trends along continuous variable, depends on compute gene trends from palantir
        returns dictionary as input for plotting functions 'plot_gene_trends' and 'plot_gene_trend_heatmap'
        adata_trends:anndata.AnnData object with trend variable and factor loadings in .obs
        factors_of_interest:list of factor key in adata_trends.obs
        trend:key for continuous variable to calculate trend over in adata_trends.obs
    trends = compute_gene_trends(adata_trends,factor_of_interest,
            trend, 
            n_jobs =1,
            imputed_layer=None)
    if type(std)==pd.core.frame.DataFrame:
        gene_trends = {trend:{'trends':trends[0],'std':std}}
    else:
        gene_trends = {trend:{'trends':trends[0],'std':std[0]}}
    return gene_trends
'''
    
def compute_factor_trends(adata_trends, factor_of_interest, trend,imputed_layer=None):
    '''
        compute factor trends along continuous variable, depends on compute gene trends from palantir
        returns dictionary as input for plotting functions 'plot_gene_trends' and 'plot_gene_trend_heatmap'
        adata_trends:anndata.AnnData object with trend variable and factor loadings in .obs
        factors_of_interest:list of factor key in adata_trends.obs
        trend:key for continuous variable to calculate trend over in adata_trends.obs
    '''
    trends,std= compute_gene_trends(adata_trends,factor_of_interest,
            trend, 
            n_jobs =1,
            imputed_layer=imputed_layer)
    if type(std)==pd.core.frame.DataFrame:
        gene_trends = {trend:{'trends':trends,'std':std}}
    else:
        gene_trends = {trend:{'trends':trends,'std':std[0]}}
    return gene_trends



def plot_gene_trends(gene_trends, genes=None):
    """ Plot the gene trends: each gene is plotted in a different panel
    :param: gene_trends: Results of the compute_marker_trends function
    from github.com/dpeerlab/Palantir
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    # Branches and genes
    branches = list(gene_trends.keys())
    colors = pd.Series(
        sns.color_palette("Set2", len(branches)).as_hex(), index=branches
    )
    if genes is None:
        genes = gene_trends[branches[0]]["trends"].index

    # Set up figure
    fig = plt.figure(figsize=[7, 3 * len(genes)])
    for i, gene in enumerate(genes):
        ax = fig.add_subplot(len(genes), 1, i + 1)
        for branch in branches:
            trends = gene_trends[branch]["trends"]
            stds = gene_trends[branch]["std"]
            ax.plot(
                trends.columns, trends.loc[gene, :], color=colors[branch], label=branch
            )
            ax.set_xticks([0, 1])
            ax.fill_between(
                trends.columns,
                trends.loc[gene, :] - stds.loc[gene, :],
                trends.loc[gene, :] + stds.loc[gene, :],
                alpha=0.1,
                color=colors[branch],
            )
            ax.set_title(gene)
        # Add legend
        if i == 0:
            ax.legend()

    sns.despine()
    
    
def plot_gene_trend_heatmaps(gene_trends,cmap='vlag'):
    """ Plot the gene trends on heatmap: a heatmap is generated or each branch
    :param: gene_trends: Results of the compute_marker_trends function
    from github.com/dpeerlab/Palantir
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    # Plot height
    branches = list(gene_trends.keys())
    genes = gene_trends[branches[0]]["trends"].index
    height = 0.7 * len(genes) * len(branches)

    #  Set up plot
    fig = plt.figure(figsize=[7, height])
    for i, branch in enumerate(branches):
        ax = fig.add_subplot(len(branches), 1, i + 1)

        # Standardize the matrix
        mat = gene_trends[branch]["trends"]
        mat = pd.DataFrame(
            StandardScaler().fit_transform(mat.T).T,
            index=mat.index,
            columns=mat.columns,
        )
        sns.heatmap(mat, xticklabels=False, ax=ax, cmap=cmap)
        ax.set_title(branch, fontsize=12)


def plot_loading_v_receptor(adata_plot, factor, receptor, celltype, use_imputed=True,s=3):
    'plot correlation spade factor vs receptor expression'
    import seaborn as sns
    import matplotlib.pyplot as plt
    adata_plot  = randomize_cells(adata_plot)

    sns.set_theme(style="white")

    if use_imputed:
        receptor_exp = np.array(adata_plot[:,receptor].layers['imputed']).flatten()
    else:
        receptor_exp = np.array(adata_plot[:,receptor].X.todense()).flatten()
    sns.relplot(x=receptor_exp, y=adata_plot.obs[factor], hue=adata_plot.obs[celltype],
            sizes=(40, 400), alpha=.5, palette="muted",
            height=6,s=s)
    
    title_corr = scipy.stats.spearmanr(receptor_exp, np.array(adata_plot.obs[factor]), axis=0, nan_policy='propagate')
    plt.xlabel(receptor)
    plt.title(title_corr)
    
    
    
#######################################statistical tests ###########################################################


def compare_factors(df, data_column, matching_variable, comparison_variable, comparison_levels,test='wilcoxon',
                         zero_method='wilcox', correction=False, 
                          alternative='two-sided',  axis=0, 
                          nan_policy='propagate'):
    '''
    compute wilcoxon matched pairs signed rank on pandas dataframe
    df: pandas.DataFrame e.g. containing aggregated data_columns per patient
    data_column: column name storing the data to be compared
    matching_variable: column name in dataframe where categorical variable for matching is stored e.g. patient
    comparison_variable: column name in dataframe where variable to be compared is stored
    comparison_levels: list containing 2 levels of comparison variable to be compared
    variables from scipy.stats.wilcoxon: zero_method='wilcox', correction=False, alternative='two-sided',  axis=0, 
    nan_policy='propagate'
    '''
    from scipy.stats import wilcoxon
    import scipy
    df_pre = df[df[comparison_variable]==comparison_levels[0]].set_index(matching_variable).dropna(axis=0)
    df_on = df[df[comparison_variable]==comparison_levels[1]].set_index(matching_variable).dropna(axis=0)
    shared_patients = set(df_on.index).intersection(set(df_pre.index))
    
    if test=='wilcoxon':
        df_pre = df_pre.reindex(shared_patients)
        df_on = df_on.reindex(shared_patients)
        print('pre and on patients identical',list(df_pre.index)==list(df_on.index))
    a = df_pre[data_column]
    b= df_on[data_column]
    if test=='wilcoxon':
        result =  wilcoxon(a, b, zero_method=zero_method, correction=correction, 
                         alternative=alternative,  axis=axis, 
                         nan_policy=nan_policy)
    elif test=='mann-whitney':
        result = scipy.stats.mannwhitneyu(a,b, use_continuity=True, 
                                          alternative='two-sided', axis=0, method='auto',
                                          nan_policy='propagate',)
    return result


def enrich_factors(gsets, factor_dict, background='hsapiens_gene_ensembl'):
    ''' 
    calculate gene set enrichment scores for gene sets in factor marker genes.
    gsets: 'dictionary containing gene set names (keys) and gene sets (values)'
    factor_dict: 'dictionary containing factor names (keys) and factor marker genes (values)
    background: 'background genes to use for gsea enrichr (accepts string for reference genome in gseapy or int for background gene number)
    ''' 
    import gseapy as gp
    #gsets= {'CD8-T_tumor-reactive-like_UP':list(input_df[input_df['gs.name']=='CD8-T_tumor-reactive-like_UP']['g.name']),
           # 'CD8-T_terminal-exhaustion':list(input_df[input_df['gs.name']=='CD8-T_terminal-exhaustion']['g.name'])}
    gsets= all_gs

    enr_df_list = []
    enr_dict = {}

    for i,v in factor_dict.items():
        glist = list(v)
        enr = gp.enrichr(gene_list=glist,
                     gene_sets=gsets,
                     background=background, # or the number of genes, e.g 20000
                     outdir=None,
                     verbose=True)
        enr.results['factor'] = len(enr.results)*[i]
        enr_df_list.append(enr.results)
        enr_dict[i] = enr
    enr_df = pd.concat(enr_df_list)

    return enr_df,enr_dict