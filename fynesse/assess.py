from .config import *

from . import access

from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

"""
FUNCTIONS FOR THE MAIN ADS ASSESSMENT
"""

# -----------------------------
# TASK 1
# 2.0) visualize data
# -----------------------------

# DONE
def preprocess_osm_features_by_frequency_geo_coa(osm_freq_by_area_df, variance_thresholding=None, standard_scaling=False):
    """
    Preprocess and prepare osm features by frequency for plotting.

    NB: we use variance thresholding here as a way to limit to the top features to plot!

    :param osm_freq_by_area_df
    :param variance_thresholding: either None or the number of top variant features t include
    :param standard_scaling: whether or not to standard scale (may be useful for sone plots)
    """
    X = osm_freq_by_area_df.loc[:, (osm_freq_by_area_df != 0).any(axis=0)].drop(columns=['coa_code'])

    # if you want to standard scale the results use sklearn for ease here
    if standard_scaling:
        Xcols = X.columns
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=Xcols)

    # we use .var() to get variance of rows, and limit to the top of them!
    if variance_thresholding:
        X = X[X.var().sort_values(ascending=False).head(variance_thresholding).index]

    return X

# DONE
def plot_correlation_matrix_on_nonzero_freq_osm_features_by_geocode(
        osm_freq_by_area_df,
        response_proportion_by_area_series=None,
        figsize=(22, 18),
        xlabel="Features",
        ylabel="Features",
        title="Correlation matrix of non-zero (across all geo codes) features",
        highest_corr_features=25
    ):
    """
    Plot a correlation matrix as shown in seaborn documentation.
    :param osm_freq_by_area_df
    :param response_proportion_by_area_series: deprecated, was meant for joining response to data
    :param figsize
    :param xlabel
    :param ylabel
    :param title
    :param highest_corr_features: int, implicit variance threshold to top number of features
    """
    sns.set_theme(style="white")

    # get the correlation matrix, but drop all features that have 0 for all geo codes
    # AND drop column coa_code as it is not required in this plot
    # data = preprocess_osm_features_by_frequency_geo_coa(osm_freq_by_area_df, variance_thresholding=variance_thresholding)
    # if response_proportion_by_area_series is not None:
    #     data['response_variable'] = response_proportion_by_area_series
    # corr = data.corr()

    # get corrs for all
    data = preprocess_osm_features_by_frequency_geo_coa(osm_freq_by_area_df)
    if response_proportion_by_area_series is not None:
        data['response_variable'] = response_proportion_by_area_series
    corr = data.corr()

    # now find the highest corrs and plot these
    response_corr = corr['response_variable'].drop('response_variable')
    highest_feature_corrs = response_corr.abs().sort_values(ascending=False).head(highest_corr_features)
    top_features = list(highest_feature_corrs.index)
    top_features.append("response_variable")

    # prepare for plot
    data = data[top_features]
    corr = data.corr()

    # we can mask the ones above the diagonal AS IN SEABORN DOCUMENTATION
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=figsize)

    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    ax = sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=0.3,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": .5}
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    cols = osm_freq_by_area_df.columns
    
    plt.show()

# DONE
def elbow_method_to_get_optimal_k(
        osm_freq_by_area_df,
        standard_scaling=True,
        k_max = 12,
        xlabel="clusters",
        ylabel="WCSS",
        title="Elbow Method to find optimal K for clustering",
        figsize=(5,3),
        proposed_elbow=4
):
    """
    Use the elbow method to find optimal number of clusters before diminishing returns.
    We use WCSS (sum of squared distance between data points and their clusters) as measure.
    Approach inspired by similar plot on GeeksForGeeks.
    :param osm_freq_by_area_df
    :param standard_scaling
    :param k_max: max k to plot to
    :param xlabel
    :param ylabel
    :param title
    :param figsize
    :param proposed_elbow: plot a vertical red line where you think the elbow is for visualization.
    """
    X = preprocess_osm_features_by_frequency_geo_coa(osm_freq_by_area_df, standard_scaling=standard_scaling)

    wcss_s = []
    k_s = [k for k in range(1, k_max + 1)]

    # here we iterate over k values to cluster and find wcss val
    for k in k_s:
        kmeans = KMeans(n_clusters=k, init='k-means++')
        kmeans.fit(X)
        wcss_s.append(kmeans.inertia_)

    # plot wcss values against number of clusters
    data = pd.DataFrame({'k_s': k_s, 'wcss_s': wcss_s})
    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.lineplot(data=data, x='k_s', y='wcss_s')

    # plot what you think is the elbow as well
    ax.axvline(proposed_elbow, color='red', linestyle='--', label='proposed elbow')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

    ax.set_xticks(k_s)
    ax.grid(color='gray', linestyle='--', alpha=0.4)

    plt.show()

# DONE
def plot_pca_visualization(
    osm_freq_by_area_df,
    _with_clusters_kmeans = 2,
    figsize=(11, 9),
    xlabel="pca_1",
    ylabel="pca_2",
    title="PCA visualization on non-zero features with K-Means clustering",
    logarithmic_axes=True,
    standard_scaling=True
):
    """
    Plot a visualization of the clusters when PCA is conducted, inspired by the lectures.
    Also inspired by lectures - use K-Means++ for better centroid initialization (less random).
    :param osm_freq_by_area_df
    :param _with_clusters_kmeans: number of clusters (should get this from elbow method), or False
    :param figsize
    :param xlabel
    :param ylabel
    :param title
    :param logarithmic_axes; bool whether or not to scale axes logarithmically
    :param standard_scaling: bool whether or not to scale data (recommended before PCA)
    """
    X = preprocess_osm_features_by_frequency_geo_coa(osm_freq_by_area_df, standard_scaling=standard_scaling)

    # only for 2 dimensions
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(X)
    pca_df = pd.DataFrame(pca_data[:, :2], columns=[xlabel, ylabel])

    fig, ax = plt.subplots(figsize=figsize)
    
    if _with_clusters_kmeans != False:
        # kmeans plot here, we use plus plus for better centroid initialization than random
        kmeans = KMeans(n_clusters=_with_clusters_kmeans, init="k-means++")
        kmeans.fit(pca_data)
        pca_df['Cluster'] = kmeans.predict(pca_data)

        ax = sns.scatterplot(data=pca_df, x=xlabel, y=ylabel, hue='Cluster')
        ax.legend()

    else:
        ax = sns.scatterplot(data=pca_df, x=xlabel, y=ylabel)

    ax.set_xlabel(f"{xlabel}{' (log axes)' if logarithmic_axes else ''}")
    ax.set_ylabel(f"{ylabel}{' (log axes)' if logarithmic_axes else ''}")
    ax.set_title(f"{title}{' (log axes)' if logarithmic_axes else ''}")
    if logarithmic_axes:
        plt.xscale("log")
        plt.yscale("log")
    ax.grid(color='gray', linestyle='--', alpha=0.4)
    plt.show()

# DONE
def scatterplot_of_most_correlated_features(
    osm_freq_by_area_df,
    response_proportion_by_area_series,
    model=LinearRegression(),
    top_features_to_plot=5,
    figsize=(8, 8),
    xlabel="feature",
    ylabel="response",
    title="Plotting correlation of most correlated features with response variable."
):
    """
    Plot the most correlated features (finding the most correlated features with R2 on submitted model).
    :param osm_freq_by_area_df
    :param response_proportion_by_area_series
    :param model: model to use to find R2 vals
    :param top_features_to_plot: i.e. choose top this many features with highest R2 value
    :param figsize
    :param xlabel
    :param ylabel
    :param title
    """
    sns.set_theme(style="white")

    # get data as one dataframe
    data = preprocess_osm_features_by_frequency_geo_coa(osm_freq_by_area_df)
    data['response_variable'] = response_proportion_by_area_series

    # calculate r2 values to find most correlated features (use this instead of corr
    # as we are using simple linear regression to observe correlation, as this will
    # likely be our end model)
    r2_values = {}
    for feature in data.columns:
        if feature != 'response_variable':
            # using a simple linear regression model i.e. line of best fit, find r2
            model.fit(data[[feature]], data['response_variable'])
            predictions = model.predict(data[[feature]])
            r2_values[feature] = r2_score(data['response_variable'], predictions)

    # get top features keying by the r2 value
    top_features = sorted(r2_values.items(), key=lambda x: x[1], reverse=True)[:top_features_to_plot]
    top_features = [f for f, _ in top_features]

    fig, ax = plt.subplots(figsize=figsize)
    for feature in top_features:
        sns.scatterplot(data=data, x=feature, y='response_variable', label=feature)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.set_xscale("log") # log scales as distributions are mostly very positively skewed

    plt.show()

# DONE
def scatterplot_of_most_correlated_20_features_separate(
    osm_freq_by_area_df,
    response_proportion_by_area_series,
    model=LinearRegression(),
    figsize=(16, 16),
    xlabel="feature",
    ylabel="response",
    title="Plotting correlation of top 10 most correlated features with response variable."
):
    """
    Plot the most correlated features (finding the most correlated features with R2 on submitted model), on 10 different subplots.
    :param osm_freq_by_area_df
    :param response_proportion_by_area_series
    :param model: model to use to find R2 vals
    :param figsize
    :param xlabel
    :param ylabel
    :param title
    """
    sns.set_theme(style="white")

    # get data as one dataframe
    data = preprocess_osm_features_by_frequency_geo_coa(osm_freq_by_area_df)
    data['response_variable'] = response_proportion_by_area_series

    # calculate r2 values to find most correlated features
    r2_values = {}
    for feature in data.columns:
        if feature != 'response_variable':
            # using a simple linear regression model i.e. line of best fit, find r2
            model.fit(data[[feature]], data['response_variable'])
            predictions = model.predict(data[[feature]])
            r2_values[feature] = r2_score(data['response_variable'], predictions)

    # get top 20 features keying by the r2 value
    top_features = sorted(r2_values.items(), key=lambda x: x[1], reverse=True)[:20]
    top_features = [f for f, _ in top_features]

    # now we plot on 20 different subplots for top 20 correlated features with response.
    fig, axs = plt.subplots(nrows=5, ncols=4, figsize=figsize)
    axs = axs.flatten()

    # iterate as recommended in seaborn
    for i in range(len(top_features)):
        feature = top_features[i]
        sns.scatterplot(data=data, x=feature, y='response_variable', label=feature, ax=axs[i])
        
        axs[i].set_xlabel(xlabel)
        axs[i].set_ylabel(ylabel)
        axs[i].set_title(f"{feature}")
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

    return top_features # we can return the features as a list as well for convenience of Assess

# DONE
def cloropleth_of_england_by_response_proportion(
    student_proportion_by_area_gdf,
    student_proportion_column_name='l15',
    zoom=None,
    radius=None
):
    """
    Plot cloropleth map of UK of student proportions.
    :param student_proportion_by_area_gdf: should contain a geometry of multipolygons to plot
    :param student_proportion_column_name: the feature to hue by
    :param zoom: optional lat long tuple centre to zoom around
    :param radius: optional radiuz of zoom
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    student_proportion_by_area_gdf.plot(column=student_proportion_column_name, ax=ax, legend=True, edgecolor='grey', linewidth=0.2, cmap='coolwarm')
    # extremely simple with built in geopandas plotting!
    
    # we can zoom using a simple bounding box around a centre (entire uk plot may be too congested)
    if zoom is not None:
        latitude, longitude = zoom
        ax.set_xlim(longitude - radius, longitude + radius)
        ax.set_ylim(latitude - radius, latitude + radius)
