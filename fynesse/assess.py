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
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import folium
import geopandas as gpd

"""
FUNCTIONS FOR THE MAIN ADS ASSESSMENT
"""

# -------------
# TASK 2
# -------------

def preprocess_census_df_task2(census_df, year=1983, variance_thresholding=None, cols_to_remove=['Party']):
    """
    Same approach as in task 1, for census data. Standard scaling not needed as we did this before.
    Same function can be used for both 2024 and 1983 census data, just specify columns to drop.
    :param historical_census_df
    :param variance_thresholding: either None or int
    """
    if year == 1983 or year == 1974:
        to_drop=['parliamentaryconstituency1983revision', 'mnemonic']
    elif year == 2024:
        census_df = census_df.rename(columns={'First party':'Party'})
        to_drop=["geography", "geography code"]
    else:
        print("error")
        return
    
    X = census_df.drop(columns=to_drop)

    if variance_thresholding:
        X = X[X.drop(columns=['responsevariable']).var().sort_values(ascending=False).head(variance_thresholding).index]

    return X.drop(columns=cols_to_remove)

def elbow_method_task2(
        historical_census_df,
        year=1983, 
        k_max=12, 
        figsize=(5,3), 
        proposed_elbow=4,
        xlabel="clusters",
        ylabel="WCSS",
        title="Elbow Method to find optimal K for clustering",
        log_yaxis=False
    ):
    """
    Plot elbow method i.e. WCSS vals against number of clusters, for given dataset.
    :param historical_census_df
    :param year
    :param k_max
    :param figsize
    :param proposed_elbow
    :param xlabel
    :param ylabel
    :param title
    :param log_yaxis
    """
    X = preprocess_census_df_task2(historical_census_df, year=year).drop(columns=['responsevariable'])

    wcss_s = []
    k_s = [k for k in range(1, k_max+1)]
    
    for k in k_s:
        kmeans = KMeans(n_clusters=k, init="k-means++")
        kmeans.fit(X)
        wcss_s.append(kmeans.inertia_)

    data = pd.DataFrame({'k_s':k_s, 'wcss_s':wcss_s})
    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.lineplot(data=data, x='k_s', y='wcss_s')
    ax.axvline(proposed_elbow, color='red', linestyle='--', label='proposed elbow')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(k_s)
    ax.set_title(title)
    if log_yaxis:
        ax.set_yscale("log")
    ax.legend()
    ax.grid(color='gray', linestyle='--', alpha=0.4)
    plt.show()

def plot_pca_visualization_task2(
    historical_census_df,
    year=1983,
    _with_clusters_kmeans=2,
    figsize=(11, 9),
    xlabel="pca_1",
    ylabel="pca_2",
    title="PCA visualization on features with K-Means clustering",
    logarithmic_axes=(True, False)
):
    """
    Plot a visualization of clusters when PCA is conducted, for task 2 data, inspured by lectures and prev task function.
    :param historical_census_df
    :param _with_clusters_kmeans
    :param figsize
    :param xlabel
    :param ylabel
    :param title
    :param logarithmic_axes: None, or two len tuple of true false corresp to xa dn y axes
    """
    X = preprocess_census_df_task2(historical_census_df, year=year).drop(columns="responsevariable")

    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(X)
    pca_df = pd.DataFrame(pca_data[:, :2], columns=[xlabel, ylabel])

    fig, ax = plt.subplots(figsize=figsize)

    if _with_clusters_kmeans != False:
        kmeans = KMeans(n_clusters=_with_clusters_kmeans, init="k-means++")
        kmeans.fit(pca_data)
        pca_df['Cluster'] = kmeans.predict(pca_data)
        ax = sns.scatterplot(data=pca_df, x=xlabel, y=ylabel, hue="Cluster")
        ax.legend()
    else:
        ax = sns.scatterplot(data=pca_df, x=xlabel, y=ylabel)
    
    ax.set_xlabel(f"{xlabel}{' (log axes)' if logarithmic_axes else ''}")
    ax.set_ylabel(f"{ylabel}{' (log axes)' if logarithmic_axes else ''}")
    ax.set_title(f"{title}{' (log axes)' if logarithmic_axes else ''}")
    if logarithmic_axes:
        if logarithmic_axes[0]:
            plt.xscale("log")
        if logarithmic_axes[1]:
            plt.yscale("log")
    ax.grid(color='gray', linestyle='--', alpha=0.4)
    plt.show()

def plot_spatial_majority_trends(gdf, opacity_col='Majority'):
    """
    Using folium and geopandas, plot a gdf of constituencies, with green for stronghold and
    orange for marginal constituencies, and have the opacity column corresponding to the feature
    submitted. Do as in the folium documentation.
    :param gdf
    :param opacity_col: feature to use as opacity
    """
    map = folium.Map(
        location=[52.898, -1.229],
        tiles='OpenStreetMap',
        zoom_start=6
    )

    copy_gdf = gdf.copy()
    copy_gdf.to_crs(epsg=4326)

    maxval = float(copy_gdf[opacity_col].max())

    for _, row in copy_gdf.iterrows():
        # colors for states
        stronghold = '#00FF00' # green
        swing = '#FFA500' # orange
        
        if row['responsevariable'] == "0":
            color = stronghold
        else:
            color = swing

        # opacity based on column given
        opacity = float(row[opacity_col])/maxval

        sim_geo = gpd.GeoSeries(row['geometry'])
        geo_j = sim_geo.to_json()
        geo_j = folium.GeoJson(
            data=geo_j, 
            style_function=lambda x, color=color, opacity=opacity: {
                'fillColor': color,
                'fillOpacity': opacity,
                'color': 'grey',
                'weight': 0.3
            }
        )
        folium.Popup(f"{row['parliamentaryconstituency1983revision']}: {'Stronghold' if row['responsevariable'] == '0' else 'Swing state'}").add_to(geo_j)
        geo_j.add_to(map)

    return map

def recursive_feature_elimination_classifier(
        historical_census_df,
        year=1983,
        n_features_to_select=15, 
        response="responsevariable", 
        estimator=DecisionTreeClassifier()
    ):
    """
    Given a dataframe with 'responsevariable' column, select the most important features
    for the given estimator. Approach inspired by TutorialsPoint.
    :param historical_census_df
    :param year
    :param n_features_to_select
    :param response
    :param estimator
    :returns: list of important suggested features after elimination
    """
    data = preprocess_census_df_task2(historical_census_df, year=year)
    X = data.drop(columns=[response])
    y = data[response]

    selector = RFE(estimator=estimator, n_features_to_select=n_features_to_select, step=1)
    selector.fit(X, y)

    rankings = list(selector.support_)
    columns = list(X.columns)
    suggested_features = []
    for i in range(len(rankings)):
        if rankings[i]:
            suggested_features.append(columns[i])
    return suggested_features

def bar_chart_feature_importances_given_model(
        historical_census_df,
        year=1983,
        features=None, 
        response='responsevariable', 
        model=RandomForestClassifier(), 
        figsize=(8,8),
        xlabel="features",
        ylabel="importances",
        title="Relative importances of top features to choose for model.",
        importance_bar=1/15
    ):
    """
    Given a model, and optional features list, fit the data to the model to show the importances of
    features relatively. Use this as threshold based elimination. Approach inspired by Machine Learning Mastery.
    :param historical_census_df
    :param year
    :param features
    :param response
    :param model
    :param figsize
    :param xlabel
    :param ylabel
    :param title
    :param importance_bar: red horizontal line to plot on bar chart, threshold
    """
    data = preprocess_census_df_task2(historical_census_df, year=year)
    if features:
        data = data[features+['responsevariable']]
    X = data.drop(columns=[response])
    y = data[response]

    model.fit(X, y)
    importance = model.feature_importances_
    recommended_features = X.columns
    data_to_plot = pd.DataFrame(data={"features": recommended_features, "importances": importance})

    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.barplot(data=data_to_plot, x="features", y="importances")
    ax.axhline(importance_bar, color='red', linestyle='--', label='importance bar')
    ax.set_xticklabels(recommended_features, rotation=90)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

    final_features = []
    for i in range(len(importance)):
        if importance[i] >= importance_bar:
            final_features.append(recommended_features[i])

    plt.show()

    return final_features

def violin_plot_of_features(
        historical_census_df,
        features,
        year=1983,
        split_on_response=True, 
        response='responsevariable', 
        figsize=(8,8)
    ):
    """
    Given data and features, plot the distributions of each feature given, to show difference
    in distribution between the categories of the response variable for each feature.
    Plot as shown in seaborn documentation.
    :param historical_census_df
    :param features
    :param year
    :param split_on_response: i.e. to plot categories separately, or as one
    :param response
    :param figsize
    """
    data = preprocess_census_df_task2(historical_census_df, year=year)[features + [response]]
    data_melted = data.melt(id_vars=[response], value_vars=features)
    fig, ax = plt.subplots(figsize=figsize)
    ax.set(ylabel=None)
    ax.set_xticklabels(features, rotation=90)
    ax.set_yticklabels([0, 0.5])
    if response:
        ax = sns.violinplot(data=data_melted, x="variable", y="value", hue=response)
    else:
        ax = sns.violinplot(data=data_melted, x="variable", y="value")
    plt.show()




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
    :returns: DataFrame of features
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
        pca_df['cluster'] = kmeans.predict(pca_data)

        ax = sns.scatterplot(data=pca_df, x=xlabel, y=ylabel, hue='cluster')
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
    radius=None,
    edgecolor='grey',
    linewidth=0.2,
    cmap='coolwarm'
):
    """
    Plot cloropleth map of UK of student proportions.
    :param student_proportion_by_area_gdf: should contain a geometry of multipolygons to plot
    :param student_proportion_column_name: the feature to hue by
    :param zoom: optional lat long tuple centre to zoom around
    :param radius: optional radiuz of zoom
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    student_proportion_by_area_gdf.plot(column=student_proportion_column_name, ax=ax, legend=True, edgecolor=edgecolor, linewidth=linewidth, cmap=cmap)
    # extremely simple with built in geopandas plotting!
    
    # we can zoom using a simple bounding box around a centre (entire uk plot may be too congested, saw this on stackoverflw)
    if zoom is not None:
        latitude, longitude = zoom
        ax.set_xlim(longitude - radius, longitude + radius)
        ax.set_ylim(latitude - radius, latitude + radius)
