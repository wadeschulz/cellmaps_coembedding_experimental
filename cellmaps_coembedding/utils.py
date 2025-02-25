import ndex2
import scipy
import random
from scipy.stats import mannwhitneyu
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from cellmaps_coembedding.muse_sc.df_utils import *
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances, cosine_similarity
import numpy as np


def plot_kde_from_df(df, kde_figure='sim_muse.png'):
    """
    Plots the KDE for similarity scores from a given DataFrame.

    Args:
    :param df: A DataFrame containing 'Similarity' values and 'Type' labels.
    :type df: pd.DataFrame
    """
    plt.figure(figsize=(8, 6))
    sns.kdeplot(data=df[df['Type'] == 'CORUM pair'], x='Similarity', color='green', label='CORUM pair')
    sns.kdeplot(data=df[df['Type'] == 'non-CORUM pair'], x='Similarity', color='grey', label='non-CORUM pair')
    plt.legend()
    plt.savefig(kde_figure, dpi=300, bbox_inches='tight')
    plt.close()


def plot_embedding_eval(results_df, embedding_eval_figure='embedding_eval.png'):
    color_map = {
        "ppi": "#659AD2",
        'image': '#c30000',
        'MUSE': "#9D85BE",
        'concat': '#C8A2C8'
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    list_keys = list(set(results_df['Embedding']))
    dataset_path_keys = [x for x in list_keys if x != 'MUSE'] + ['MUSE']

    # Iterate over datasets to plot
    muse = results_df[results_df['Embedding'] == 'MUSE']['Result'].values
    for idx, embedding in enumerate(dataset_path_keys):
        subset_embedding = results_df[results_df['Embedding'] == embedding]
        mean_result = np.mean(subset_embedding['Result'].values)
        stdev = np.std(subset_embedding['Result'].values)
        bootstrap_mean_diff = muse - subset_embedding['Result'].values
        bootstrap_mean_diff_average = np.mean(bootstrap_mean_diff)
        # shift so null is true
        shifted = np.array([x - bootstrap_mean_diff_average for x in bootstrap_mean_diff])
        # p-value = how many differences are more extreme than actual mean difference if null true; add 1 so p-value is not 0
        pval = float((1 + len(shifted[shifted > np.abs(bootstrap_mean_diff_average)])) / (1 + len(shifted)) + (
                1 + len(shifted[shifted < -np.abs(bootstrap_mean_diff_average)])) / (1 + len(shifted)))
        sig = False
        if embedding != 'MUSE':
            if pval < 0.05:
                sig = True

        lower = mean_result - stdev
        upper = mean_result + stdev
        if embedding == 'ppi' or embedding == 'image':
            shape = '^'
        else:
            shape = 'o'
        # Plot the horizontal bars for confidence intervals in grey
        ax.hlines(2 + 0.2 * (idx - len(dataset_path_keys) / 2), lower, upper, color='grey',
                  alpha=0.3)  # Set the color to grey
        alpha = 1.0 if embedding == "MUSE" else 0.6  # Full opacity for "CO-EMBEDDING", reduced for others
        ax.scatter(mean_result, 2 + 0.2 * (idx - len(dataset_path_keys) / 2), alpha=alpha, color=color_map[embedding],
                   marker=shape, label=embedding, s=300)
        if sig:
            ax.text(mean_result, 2 + 0.2 * (idx - len(dataset_path_keys) / 2), '*', fontsize=15, color='black',
                    ha='center', va='center')

    plt.yticks([])  # Hides y-axis ticks
    # Adjust the legend
    plt.xlabel('Enrichment')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlim((0, 1))
    # Save and show the plot
    plt.tight_layout()
    plt.savefig(embedding_eval_figure, dpi=300, bbox_inches='tight')
    plt.close()


def get_embedding_eval_data(dataset_paths, edges, num_samplings, num_edges, kde_file, eval_file, common_proteins=None):
    dataframes = {dataset: pd.read_table(dataset_paths[dataset], index_col=0) for dataset in dataset_paths.keys()}
    if 'ppi' in dataset_paths.keys() and 'image' in dataset_paths.keys():
        df1 = dataframes['ppi']
        df2 = dataframes['image']
        df_concatenated = pd.concat([df1, df2], axis=1, ignore_index=True)
        dataframes['concat'] = df_concatenated

    # Placeholder for results and eval set sizes
    results = []

    index_sets = [set(df.index) for df in dataframes.values()]
    if common_proteins is None:
        common_proteins = set.intersection(*index_sets)

    eval_set = list(common_proteins.intersection(edges.index))

    # Calculate for each latent embedding
    for modality_name, df in dataframes.items():
        eval_set = list(common_proteins.intersection(edges.index))
        filtered_edges = edges.loc[eval_set, eval_set]
        if df.shape[0] == df.shape[1]:
            sim_df = df.loc[eval_set, eval_set]
        else:
            sim_df = cosine_similarity_scaled(df.loc[eval_set])
        filtered_df = sim_df.loc[eval_set, eval_set]
        upper_tri_indices = np.triu_indices_from(filtered_df, k=1)
        sim_df_values = filtered_df.values[upper_tri_indices]
        filtered_edges_values = filtered_edges.values[upper_tri_indices]

        for sampling in np.arange(num_samplings):
            true = random.choices(sim_df_values[filtered_edges_values == 1], k=num_edges)
            false = random.choices(sim_df_values[filtered_edges_values == 0], k=num_edges)
            u, p = mannwhitneyu(true, false, alternative='greater')
            effect_size = (u / (len(true) * len(false)) - 0.5) * 2
            results.append({'Embedding': modality_name, 'Result': effect_size})

        if modality_name == 'MUSE':
            true = sim_df_values[filtered_edges_values == 1]
            false = sim_df_values[filtered_edges_values == 0]
            # u, p = mannwhitneyu(true, false, alternative='greater')
            # print(p)
            kde_data = pd.DataFrame({
                'Similarity': np.concatenate([true, false]),
                'Type': ['CORUM pair'] * len(true) + ['non-CORUM pair'] * len(false)
            })
            kde_data.to_csv(kde_file, index=False)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    results_df.to_csv(eval_file, index=False)


def generate_embedding_evaluation_figures(coembedding, ppi=None, image=None, outdir='.', num_samplings=1000,
                                          num_edges=1000):

    dataset_paths = {'MUSE': os.path.join(coembedding, 'coembedding_emd.tsv')}
    if ppi:
        dataset_paths['ppi'] = os.path.join(ppi, 'ppi_emd.tsv')
    if image:
        dataset_paths['image'] = os.path.join(image, 'image_emd.tsv')

    # CORUM
    corum_file = os.path.join(os.path.dirname(__file__), 'corum_M.tsv')
    corum_M = pd.read_csv(corum_file, sep='\t', index_col=0)

    kde_file = os.path.join(outdir, 'sim_muse_data.csv')
    eval_file = os.path.join(outdir, 'embedding_eval.csv')
    get_embedding_eval_data(dataset_paths, corum_M, num_samplings=num_samplings, num_edges=num_edges, kde_file=kde_file,
                            eval_file=eval_file)

    kde_data = pd.read_csv(kde_file, index_col=0)
    plot_kde_from_df(kde_data)

    embedding_eval_df = pd.read_csv(eval_file)
    plot_embedding_eval(embedding_eval_df)
