#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 14:22:47 2021

@author: jocelynreahl
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
import matplotlib.transforms as transforms
from mpl_toolkits.axes_grid1 import make_axes_locatable


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.
    https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html
    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    ---------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def run_PCA_fit_transform(dataset, tex):
    '''
    Run PCA fit-transform on dataset (fits the PCA model to the dataset and
    apply dimensionality reduction to dataset).
    ----------
    dataset = pandas dataframe
    tex = list of microtexture abbreviations to use for PCA ordination
    
    Returns
    ----------
    pca_df = pandas dataframe with coordinates for each sample on each PC axis
    pca = pca model object
    '''
    data = dataset.loc[:, tex]
    scaled_data = preprocessing.scale(data)
    pca = PCA()
    pca_fit_transform = pca.fit_transform(scaled_data)
    per_var = np.round(pca.explained_variance_ratio_*100, decimals=2)
    components = ['PC' + str(x) for x in range(1, len(per_var)+1)]
    pca_df = pd.DataFrame(pca_fit_transform, columns=components)
    pca_df['transport'] = list(dataset['transport'])
    pca_df['author'] = list(dataset['author'])
    pca_df['marker'] = list(dataset['marker'])
    pca_df['color'] = list(map(lambda s: s.replace('\ufeff', ''),
                               dataset['transportcolor']))
    return pca_df, pca


def run_PCA_transform(dataset, tex, pca):
    '''
    Run PCA transform on dataset (applies existing dimensionality reduction
    from initial pca.fit_transform() to new dataset).
    ----------
    dataset = pandas dataframe
    tex = list of microtexture abbreviations to use for PCA ordination
    pca = pca model object from run_PCA_fit_transform()
    
    Returns
    ----------
    pca_df = pandas dataframe with coordinates for each sample on each PC axis
    '''
    if 'pf' in set(tex):
        dataset = dataset[dataset['author'] != 'Sweet_2010']
        data = dataset.loc[:, tex]
    else:
        data = dataset.loc[:, tex]
    scaled_data = preprocessing.scale(data)
    pca_transform = pca.transform(scaled_data)
    per_var = np.round(np.round(pca.explained_variance_ratio_*100,
                                decimals=1))
    components = ['PC' + str(x) for x in range(1, len(per_var)+1)]
    pca_df = pd.DataFrame(pca_transform, columns=components)
    pca_df['transport'] = list(dataset['transport'])
    pca_df['author'] = list(dataset['author'])
    pca_df['marker'] = list(dataset['marker'])
    pca_df['color'] = list(map(lambda s: s.replace('\ufeff', ''),
                               dataset['transportcolor']))
    return pca_df


def get_pervar_loadings(pca, tex):
    per_var = np.round(pca.explained_variance_ratio_*100, decimals=2)
    components = ['PC' + str(x) for x in range(1, len(per_var)+1)]
    loading_scores = pd.DataFrame(list(zip(tex, pca.components_[0],
                                           pca.components_[1],
                                           pca.components_[2])),
                                      index=None,
                                      columns=['microtextures', 'PC1', 'PC2',
                                               'PC3'])
    # Print out Variance for PC1-PC3 in the console:
    print('Principal Component Variance:')
    print(per_var)
    print('Sum of PC1 = ' + str(per_var[0]))
    print('Sum of PC1 + PC2 = ' + str(per_var[0] + per_var[1]))
    print('Sum of PC1 + PC2 + PC3 = ' + str(per_var[0] +
                                            per_var[1] +
                                            per_var[2]))
    print('Microtextural Loading Scores:')
    print(loading_scores)
    print('Sorted Scores:')
    PC1sorted = PCAscore_sorter(pca, tex, 1)
    PC2sorted = PCAscore_sorter(pca, tex, 2)
    PC3sorted = PCAscore_sorter(pca, tex, 3)
    print('PC1:')
    print(PC1sorted)
    print('PC2:')
    print(PC2sorted)
    print('PC3:')
    print(PC3sorted)
    
    return per_var, components, loading_scores


def sortorder(pca_df, trans_list, palette_list):
    pca_df.transport = pca_df.transport.astype('category')
    pca_df.transport.cat.set_categories(trans_list, inplace=True)
    pca_df = pca_df.sort_values(['transport'])
    return pca_df


def textplot(ax, tex, x, y, PCx, PCy, label):
    """
    Plot text on PCA plots so that microtexture abbreviations do not overlap
    with arrows.
    ----------
    ax = axis object (e.g. ax1, ax2, etc.)
    tex = microtextures used to make PCA ordination
    x = float; x coordinate of vector
    y = float; y coordinate of vector
    PCx = integer; indicate principal component that x-axis is using
          PC1 = 0, PC2 = 1, PC3 = 2
    PCy = integer; indicate principal component that y-axis is using
          PC1 = 0, PC2 = 1, PC3 = 2
    label = string; text label
    """
    adj = 0.01  # Adjustment factor
    s = 24
    p = len(tex)
    b_k = np.zeros(p)
    total = np.zeros(p)
    for i in range(p):
        for j in range(i+1, p+1):
            total[i] += 1/j
        b_k[i] = (1/p)*total[i]
    if np.square(x) > b_k[PCx] or np.square(y) > b_k[PCy]:
        if x > 0 and y > 0:
            ax.text(x, y+adj, label, size=s, weight='black')
        if x > 0 and y < 0:
            ax.text(x, y-adj, label, size=s, weight='black')
        if x < 0 and y < 0:
            if x < -0.6:
                ax.text(-0.6, y+adj, label, size=s, weight='black')
            else:
                ax.text(x-(adj*5), y-(adj*2), label, size=s, weight='black')
        if x < 0 and y > 0:
            if x < -0.55:
                ax.text(-0.55, y+adj, label, size=s, weight='black')
            else:
                ax.text(x-(adj*5), y+adj, label, size=s, weight='black')
    else:
        if x > 0 and y > 0:
            ax.text(x, y+adj, label, size=s)
        if x > 0 and y < 0:
            ax.text(x, y-adj, label, size=s)
        if x < 0 and y < 0:
            if x < -0.6:
                ax.text(-0.6, y+adj, label, size=s)
            else:
                ax.text(x-(adj*5), y-(adj*2), label, size=s)
        if x < 0 and y > 0:
            if x < -0.55:
                ax.text(-0.55, y+adj, label, size=s)
            else:
                ax.text(x-(adj*5), y+adj, label, size=s)


def PCAscore_sorter(pca, tex, n):
    '''
    Sort PCA loadings by loading score.
    ----------
    pca = pca model object
    tex = list of microtexture abbreviations; same as used for PCA ordination
    n = PC axis; ex. 1 = PC1, 2 = PC2, etc.
    
    Returns
    ----------
    PCsorted = pandas Series with sorted loadings
    '''
    PCloadings = pd.Series(pca.components_[n-1], index=tex)
    PCsorted = PCloadings.sort_values(ascending=False)
    # PCsorted = PCsorted.index.values
    return PCsorted


def plot_trainingdata(dataframe, PCx, PCy, ax, background=False):
    for k in range(len(dataframe)):
        fk = dataframe['color'].loc[k]
        mk = dataframe['marker'].loc[k]
        xk = dataframe[PCx].loc[k]
        yk = dataframe[PCy].loc[k]
        if background == False:
            ax.scatter(xk, yk, marker=mk, facecolors=fk, edgecolors='k',
                       s=200, alpha=1, linewidths=1)
        else:
            ax.scatter(xk, yk, marker=mk, facecolors=fk, edgecolors=fk,
                       s=200, alpha=0.5, linewidths=1)
    # Plot associated confidence ellipses
    colors = ['#D55E00', '#0072B2', '#F0E442']
    for c in colors:
        data = dataframe[dataframe['color'] == c]
        confidence_ellipse(data[PCx], data[PCy], ax, n_std=2,
                           facecolor='none', edgecolor=c, alpha=1, lw=2)


def plot_testdata(dataframe, PCx, PCy, ax):
    for k in range(len(dataframe)):
        fk = dataframe['color'].loc[k]
        mk = dataframe['marker'].loc[k]
        xk = dataframe[PCx].loc[k]
        yk = dataframe[PCy].loc[k]
        ax.scatter(xk, yk, marker=mk, facecolors=fk, edgecolors='k',
                   s=200, alpha=1, linewidths=1)


def plot_loadings(loading_scores, PCx, PCy, ax, tex):
    for labels, PC1, PC2 in zip(loading_scores['microtextures'],
                                loading_scores[PCx],
                                loading_scores[PCy]):
        ax.arrow(0, 0, PC1, PC2, color='#000000',
                                       length_includes_head=True,
                                       head_width=0.02, overhang=0.05)
        textplot(ax, tex, PC1, PC2, 0, 1, labels)


def PCAplot_3x3(dataframe, tex, label, groupby='transport'):
    '''
    Perform PCA and plot biplots of PCA data and loadings in 3x3 order.
    ----------
    dataframe = pandas Dataframe; use data from ALLDATA.csv
    tex = list of microtexture abbreviations to use for PCA ordination
    label = str; file name with no extension; directs to Figures folder
    groupby = str; 'transport' = samples are grouped by transport mode,
              'climate' = samples are grouped by ice presence/absence.
    '''
    # Define "train" and "test" datasets from dataframe:
    train = dataframe[(dataframe['relage'] == 'Active')]
    pca_df_tr, pca = run_PCA_fit_transform(train, tex) # training PCA
    if 'pf' in set(tex):
        test = dataframe[(dataframe['relage'] != 'Active') &
                           (dataframe['author'] != 'Sweet_2010')]
    else:
        test = dataframe[(dataframe['relage'] != 'Active')]
    pca_df_te = run_PCA_transform(test, tex, pca)
    
    # Get percent variance and loading scores from fitted PCA model:
    per_var, components, loading_scores = get_pervar_loadings(pca, tex)
    
    # Generate a Scree Plot for the PCA training plot
    plt.bar(x=range(1, len(per_var)+1), height=per_var,
            tick_label=components)
    plt.ylabel('Percentage of Explained Variance [%]')
    plt.xlabel('Principal Component')
    
    # Plot PCA ordination w/reference first, then sample
    fig, ax = plt.subplots(3, 3, figsize=(20, 20)) # Set up axes
    ls = 24
    for i in range(3):
        for j in range(3):
            ax[i, j].tick_params(axis='both', direction='in', which='major',
                             top=True, labeltop=False, right=True,
                             labelright=False, left=True, bottom=True,
                             labelsize=ls)
            if j == 0:
                # Set x and y limits for whole column 0:
                ax[i, j].set_xlim(-5, 6)
                ax[i, j].set_ylim(-5, 6)
                if i == 0:
                    # Add axes labels and sub-figure label:
                    ax[i, j].set_xlabel('PC1', size=ls)
                    ax[i, j].set_ylabel('PC2', size=ls)
                    ax[i, j].text(-4.5, 4.5, 'A1', size=40)
                    # Add rectangular label at top of column 0:
                    ax[i, j].add_patch(Rectangle((-5, 6+0.5), 11, 1,
                                                 clip_on=False, fill=True,
                                                 facecolor='#648FFF',
                                                 edgecolor='w'))
                    ax[i, j].text(0.5, 7, 'Modern Samples', size=ls, c='w',
                                  horizontalalignment='center',
                                  verticalalignment='center', weight='bold')
                    # Add rectangular label at left of row 0:
                    ax[i, j].add_patch(Rectangle((-5-3.1, -5), 1, 11,
                                                 clip_on=False, fill=True,
                                                 facecolor='#648FFF',
                                                 edgecolor='w'))
                    ax[i, j].text(-7.5, 0.5, 'PC1 v. PC2', size=ls, c='w',
                                  horizontalalignment='center',
                                  verticalalignment='center', weight='bold',
                                  rotation=90)
                    # Plot training data for PC1 vs PC3:
                    plot_trainingdata(pca_df_tr, 'PC1', 'PC2', ax[i, j])
                elif i == 1:
                    # Add axes labels and sub-figure label:
                    ax[i, j].set_xlabel('PC1', size=ls)
                    ax[i, j].set_ylabel('PC3', size=ls)
                    ax[i, j].text(-4.5, 4.5, 'B1', size=40)
                    # Add rectangular label at left of row 1:
                    ax[i, j].add_patch(Rectangle((-5-3.1, -5), 1, 11,
                                                 clip_on=False, fill=True,
                                                 facecolor='#648FFF',
                                                 edgecolor='w'))
                    ax[i, j].text(-7.5, 0.5, 'PC1 v. PC3', size=ls, c='w',
                                  horizontalalignment='center',
                                  verticalalignment='center', weight='bold',
                                  rotation=90)
                    # Plot training data for PC1 vs PC3:
                    plot_trainingdata(pca_df_tr, 'PC1', 'PC3', ax[i, j])
                elif i == 2:
                    # Add axes labels and sub-figure label:
                    ax[i, j].set_xlabel('PC2', size=ls)
                    ax[i, j].set_ylabel('PC3', size=ls)
                    ax[i, j].text(-4.5, 4.5, 'C1', size=40)
                    # Add rectangular label at left of row 2:
                    ax[i, j].add_patch(Rectangle((-5-3.1, -5), 1, 11,
                                                 clip_on=False, fill=True,
                                                 facecolor='#648FFF',
                                                 edgecolor='w'))
                    ax[i, j].text(-7.5, 0.5, 'PC2 v. PC3', size=ls, c='w',
                                  horizontalalignment='center',
                                  verticalalignment='center', weight='bold',
                                  rotation=90)
                    # Plot training data for PC2 vs PC3:
                    plot_trainingdata(pca_df_tr, 'PC2', 'PC3', ax[i, j])
            elif j == 1:
                # Set x and y limits for whole column 1:
                ax[i, j].set_xlim(-0.6, 0.6)
                ax[i, j].set_ylim(-0.7, 0.7)
                if i == 0:
                    # Add labels and sub-figure label:
                    ax[i, j].set_xlabel('PC1', size=ls)
                    # ax[i, j].set_ylabel('PC2', size=ls)
                    ax[i, j].text(-0.545, 0.51, 'A2', size=40)
                    # Add rectangular label at top of column 1:
                    ax[i, j].add_patch(Rectangle((-0.6, 0.7+0.064), 1.2, 0.127,
                                                 clip_on=False, fill=True,
                                                 facecolor='#648FFF',
                                                 edgecolor='w'))
                    ax[i, j].text(0, 0.7+0.064+(0.127/2), 'PC Loadings',
                                  size=ls, c='w',
                                  horizontalalignment='center',
                                  verticalalignment='center', weight='bold')
                    # Plot the loadings for PC1 and PC2:
                    plot_loadings(loading_scores, 'PC1', 'PC2', ax[i, j], tex)
                elif i == 1:
                    # Add labels and sub-figure label:
                    ax[i, j].set_xlabel('PC1', size=ls)
                    # ax[i, j].set_ylabel('PC3', size=ls)
                    ax[i, j].text(-0.545, 0.51, 'B2', size=40)
                    # Plot the loadings for PC1 and PC3:
                    plot_loadings(loading_scores, 'PC1', 'PC3', ax[i, j], tex)
                elif i == 2:
                    # Add labels and sub-figure label:
                    ax[i, j].set_xlabel('PC2', size=ls)
                    # ax[i, j].set_ylabel('PC3', size=ls)
                    ax[i, j].text(-0.545, 0.51, 'C2', size=40)
                    # Plot the loadings for PC2 and PC3:
                    plot_loadings(loading_scores, 'PC2', 'PC3', ax[i, j], tex)
            elif j == 2:
                ax[i, j].set_xlim(-5, 6)
                ax[i, j].set_ylim(-5, 6)
                if i == 0:
                    # Add labels and sub-figure label:
                    ax[i, j].set_xlabel('PC1', size=ls)
                    # ax[i, j].set_ylabel('PC2', size=ls)
                    ax[i, j].text(-4.5, 4.5, 'A3', size=40)
                    # Add rectangular label at top of column 2:
                    ax[i, j].add_patch(Rectangle((-5, 6+0.5), 11, 1,
                                                 clip_on=False, fill=True,
                                                 facecolor='#648FFF',
                                                 edgecolor='w'))
                    ax[i, j].text(0.5, 7, 'Ancient Samples', size=ls, c='w',
                                  horizontalalignment='center',
                                  verticalalignment='center', weight='bold')
                    # Plot training data for PC1 and PC2 as background:
                    plot_trainingdata(pca_df_tr, 'PC1', 'PC2', ax[i, j],
                                      background=True)
                    # Plot test data for PC1 and PC2:
                    plot_testdata(pca_df_te, 'PC1', 'PC2', ax[i, j])
                elif i == 1:
                    # Add axes labels and sub-figure label:
                    ax[i, j].set_xlabel('PC1', size=ls)
                    # ax[i, j].set_ylabel('PC3', size=ls)
                    ax[i, j].text(-4.5, 4.5, 'B3', size=40)
                    # Plot training data for PC1 and PC3 as background:
                    plot_trainingdata(pca_df_tr, 'PC1', 'PC3', ax[i, j],
                                      background=True)
                    # Plot test data for PC1 and PC3:
                    plot_testdata(pca_df_te, 'PC1', 'PC3', ax[i, j])
                elif i == 2:
                    # Add axes labels and figure sub-labels:
                    ax[i, j].set_xlabel('PC2', size=ls)
                    # ax[i, j].set_ylabel('PC3', size=ls)
                    ax[i, j].text(-4.5, 4.5, 'C3', size=40)
                    # Plot training data for PC2 and PC3 as background:
                    plot_trainingdata(pca_df_tr, 'PC2', 'PC3', ax[i, j],
                                      background=True)
                    # Plot test data for PC2 and PC3:
                    plot_testdata(pca_df_te, 'PC2', 'PC3', ax[i, j])
    plt.savefig('Figures/PCA-' + label.upper() + '.jpg', dpi=300)
    plt.show()