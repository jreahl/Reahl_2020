#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 10:10:05 2020

@author: jocelynreahl
"""

# Import packages:
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Ellipse, Rectangle
import matplotlib.transforms as transforms
from matplotlib.lines import Line2D

# Import ALLDATA.csv
lower = pd.read_csv('Data_CSV/ALLDATA_NARTISS-LOWER.csv')
middle = pd.read_csv('Data_CSV/ALLDATA_NARTISS-MIDDLE.csv')
upper = pd.read_csv('Data_CSV/ALLDATA_NARTISS-UPPER.csv')
bounds = [lower, middle, upper]

# Define microtexture sets to use for PCA:

# tex_allpossible = ['af', 'as', 'bb', 'cf', 'ff', 'ls', 'saf', 'slf', 'up',  # Polygenetic
#                    'er', 'vc',  # Percussion
#                    'crg', 'cg', 'dt', 'sg',  # High-stress
#                    'de', 'pf',  # Chemicals
#                    'low', 'med', 'high']  # Relief
tex_allauthors = ['as', 'cf', 'cg', 'er', 'ls', 'pf', 'saf', 'slf', 'vc',
                  'low', 'med', 'high']
tex_mechanical = ['as', 'cf', 'cg', 'er', 'ls', 'saf', 'slf', 'vc', 'low',
                  'med', 'high']


class RepackagedData:
    '''
    Class to reformat input data for the PCA plotting functions.
    '''
    def __init__(self, data, tex):
        self.data = data.loc[:, tex]
        self.transportcolor = list(map(lambda s: s.replace('\ufeff', ''),
                                   data.transportcolor))
        self.climatecolor = list(map(lambda s: s.replace('\ufeff', ''),
                                 data.climatecolor))
        self.marker = data.marker



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
    s = 16
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


def Sensitivityplot(datalist, tex, label):
    '''
    Plot lower (column 1), middle (column 2), and upper (column 3) bounds of
    Nartišs and Kalińska-Nartiša (2017) ancient data when transformed into
    modern PCA space.
    ----------
    datalist = list of pandas Dataframes; use list of lower, middle, and upper
               Nartišs and Kalińska-Nartiša (2017) data
    tex = list of microtexture abbreviations to use for PCA ordination
    label = str; file name with no extension; directs to Figures folder
    '''
    # Define "reference" and "sample" datasets and perform pca.fit_transform()
    # and pca.transform() on the reference (modern) and sample (ancient)
    # datasets, respectively.
    modern_list = [] # empty list to store modern fit-transformed PCA data
    pca_list = [] # empty list to store PCA model objects
    ancient_list = [] # empty list to store ancient transformed PCA data
    for d in datalist:
        reference = d[d['relage'] == 'Active']
        sample = d[d['relage'] != 'Active']
        pca_df_m, pca = run_PCA_fit_transform(reference, tex)
        pca_df_a = run_PCA_transform(sample, tex, pca)
        modern_list.append(pca_df_m)
        pca_list.append(pca)
        ancient_list.append(pca_df_a)
    
    # Plot PCA ordination w/reference first, then sample
    fig, ax = plt.subplots(3, 3, figsize=(20, 20)) # Set up axes
    for i in range(3):
        for j in range(3):
            ax[i, j].tick_params(axis='both', which='major', top=True,
                                 labeltop=False, right=True, labelright=False,
                                 labelsize=14)
            ax[i, j].set_xlim(-5, 6)
            ax[i, j].set_ylim(-5, 6)
            data_m = modern_list[j]
            data_a = ancient_list[j]
            if i == 0:
                ax[i, j].set_xlabel('PC1', size=20)
                ax[i, j].text(-4.5, 4.5, 'A'+str(j+1), size=40)
                # for k in range(len(data_m)):
                #         fk = data_m['color'].loc[k]
                #         mk = data_m['marker'].loc[k]
                #         xk = data_m['PC1'].loc[k]
                #         yk = data_m['PC2'].loc[k]
                #         ax[i, j].scatter(xk, yk, marker=mk, facecolors=fk,
                #                           edgecolors=fk, s=200, alpha=0.5,
                #                           linewidths=1)
                colors = ['#D55E00', '#0072B2', '#F0E442']
                for c in colors:
                    trans_group = data_m[data_m['color'] == c]
                    confidence_ellipse(trans_group['PC1'],
                                           trans_group['PC2'], ax[i, j],
                                           n_std=2, facecolor='none',
                                           edgecolor=c, alpha=1, lw=2)
                for k in range(len(data_a)):
                    fk = data_a['color'].loc[k]
                    mk = data_a['marker'].loc[k]
                    xk = data_a['PC1'].loc[k]
                    yk = data_a['PC2'].loc[k]
                    if data_a['author'].loc[k] == 'Nartiss_2017':
                        sk = 1000
                    elif data_a['author'].loc[k] != 'Nartiss_2017':
                        sk = 200
                    ax[i, j].scatter(xk, yk, marker=mk, facecolors=fk,
                                         edgecolors='k', s=sk, alpha=1,
                                         linewidths=1)
                if j == 0:
                    ax[i, j].set_ylabel('PC2', size=20)
                    ax[i, j].add_patch(Rectangle((-5, 6+0.5), 11, 1,
                                                 clip_on=False, fill=True,
                                                 facecolor='#648FFF',
                                                 edgecolor='w'))
                    ax[i, j].text(0.5, 7, 'Lower Bound', size=20, c='w',
                                  horizontalalignment='center',
                                  verticalalignment='center', weight='bold')
                    ax[i, j].add_patch(Rectangle((-5-3, -5), 1, 11,
                                                 clip_on=False, fill=True,
                                                 facecolor='#648FFF',
                                                 edgecolor='w'))
                    ax[i, j].text(-7.5, 0.5, 'PC1 v. PC2', size=20, c='w',
                                  horizontalalignment='center',
                                  verticalalignment='center', weight='bold',
                                  rotation=90)
                if j == 1:
                    ax[i, j].add_patch(Rectangle((-5, 6+0.5), 11, 1,
                                                 clip_on=False, fill=True,
                                                 facecolor='#648FFF',
                                                 edgecolor='w'))
                    ax[i, j].text(0.5, 7, 'Middle Bound', size=20, c='w',
                                  horizontalalignment='center',
                                  verticalalignment='center', weight='bold')
                if j == 2:
                    ax[i, j].add_patch(Rectangle((-5, 6+0.5), 11, 1,
                                                 clip_on=False, fill=True,
                                                 facecolor='#648FFF',
                                                 edgecolor='w'))
                    ax[i, j].text(0.5, 7, 'Upper Bound', size=20, c='w',
                                  horizontalalignment='center',
                                  verticalalignment='center', weight='bold')
            if i == 1:
                ax[i, j].set_xlabel('PC1', size=20)
                ax[i, j].text(-4.5, 4.5, 'B'+str(j+1), size=40)
                # for k in range(len(data_m)):
                #         fk = data_m['color'].loc[k]
                #         mk = data_m['marker'].loc[k]
                #         xk = data_m['PC1'].loc[k]
                #         yk = data_m['PC3'].loc[k]
                #         ax[i, j].scatter(xk, yk, marker=mk, facecolors=fk,
                #                           edgecolors=fk, s=200, alpha=0.5,
                #                           linewidths=1)
                colors = ['#D55E00', '#0072B2', '#F0E442']
                for c in colors:
                    trans_group = data_m[data_m['color'] == c]
                    confidence_ellipse(trans_group['PC1'], trans_group['PC3'],
                                       ax[i, j], n_std=2, facecolor='none',
                                       edgecolor=c, alpha=1, lw=2)
                for k in range(len(data_a)):
                    fk = data_a['color'].loc[k]
                    mk = data_a['marker'].loc[k]
                    xk = data_a['PC1'].loc[k]
                    yk = data_a['PC3'].loc[k]
                    if data_a['author'].loc[k] == 'Nartiss_2017':
                        sk = 1000
                    elif data_a['author'].loc[k] != 'Nartiss_2017':
                        sk = 200
                    ax[i, j].scatter(xk, yk, marker=mk, facecolors=fk,
                                         edgecolors='k', s=sk, alpha=1,
                                         linewidths=1)
                if j == 0:
                    ax[i, j].set_ylabel('PC3', size=20)
                    ax[i, j].add_patch(Rectangle((-5-3, -5), 1, 11,
                                                 clip_on=False, fill=True,
                                                 facecolor='#648FFF',
                                                 edgecolor='w'))
                    ax[i, j].text(-7.5, 0.5, 'PC1 v. PC3', size=20, c='w',
                                  horizontalalignment='center',
                                  verticalalignment='center', weight='bold',
                                  rotation=90)
            if i == 2:
                ax[i, j].set_xlabel('PC2', size=20)
                ax[i, j].text(-4.5, 4.5, 'C'+str(j+1), size=40)
                # for k in range(len(data_m)):
                #         fk = data_m['color'].loc[k]
                #         mk = data_m['marker'].loc[k]
                #         xk = data_m['PC2'].loc[k]
                #         yk = data_m['PC3'].loc[k]
                #         ax[i, j].scatter(xk, yk, marker=mk, facecolors=fk,
                #                           edgecolors=fk, s=200, alpha=0.5,
                #                           linewidths=1)
                colors = ['#D55E00', '#0072B2', '#F0E442']
                for c in colors:
                    trans_group = data_m[data_m['color'] == c]
                    confidence_ellipse(trans_group['PC2'], trans_group['PC3'],
                                       ax[i, j], n_std=2, facecolor='none',
                                       edgecolor=c, alpha=1, lw=2)
                for k in range(len(data_a)):
                    fk = data_a['color'].loc[k]
                    mk = data_a['marker'].loc[k]
                    xk = data_a['PC2'].loc[k]
                    yk = data_a['PC3'].loc[k]
                    if data_a['author'].loc[k] == 'Nartiss_2017':
                        sk = 1000
                    elif data_a['author'].loc[k] != 'Nartiss_2017':
                        sk = 200
                    ax[i, j].scatter(xk, yk, marker=mk, facecolors=fk,
                                         edgecolors='k', s=sk, alpha=1,
                                         linewidths=1)
                if j == 0:
                    ax[i, j].set_ylabel('PC3', size=20)
                    ax[i, j].add_patch(Rectangle((-5-3, -5), 1, 11,
                                                 clip_on=False, fill=True,
                                                 facecolor='#648FFF',
                                                 edgecolor='w'))
                    ax[i, j].text(-7.5, 0.5, 'PC2 v. PC3', size=20, c='w',
                                  horizontalalignment='center',
                                  verticalalignment='center', weight='bold',
                                  rotation=90)
    plt.savefig('Figures/SENSITIVITY-TEST-' + label.upper() + '.jpg', dpi=300)
    plt.show()