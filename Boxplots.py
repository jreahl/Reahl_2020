#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 15:34:15 2020

Author: Jocelyn N Reahl
Title: BOXPLOTS
Description: Script to generate boxplots for all-textures and mechanical PCA
ordinations.
"""

# Import packages
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from sklearn import preprocessing
from sklearn.decomposition import PCA
from scipy import stats
import numpy as np

sns.set(style='white') # Set style for Seaborn plot

# Import data and split it up into modern and ancient datasets
master2 = pd.read_csv('Data_CSV/ALLDATA.csv')
modern = master2[master2['relage'] == 'Active']
ancient = master2[master2['relage'] == 'Ancient']

# Define textures used in PCA ordination 
tex_allauthors = ['as', 'cf', 'cg', 'er', 'ls', 'pf', 'saf', 'slf', 'vc',
                  'low', 'med', 'high']
tex_mechanical = ['as', 'cf', 'cg', 'er', 'ls', 'saf', 'slf', 'vc', 'low',
                  'med', 'high']


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


def calc_boxplot(dataset, ordination, groupcolumn, groupstr, PC):
    '''
    Calculate boxplot statistics for dataset.
    ----------
    dataset = pandas dataframe
    ordination = str; name of the ordination, e.g. "All Textures" or
                 "Mechanical"
    groupcolumn = str; column to group dataset by
    groupstr = str; value in column to group dataset by
    PC = str; principal component to sort by (e.g. 'PC1', 'PC2', etc.)
    '''
    d = dataset[dataset[groupcolumn] == groupstr].loc[:, PC]
    q25, q50, q75 = np.percentile(d, [25, 50, 75])
    whiskerlim = 1.5 * stats.iqr(d)
    h1 = np.min(d[d >= (q25 - whiskerlim)])
    h2 = np.max(d[d <= (q75 + whiskerlim)])
    return pd.Series([ordination, groupstr, PC, q25, q50, q75, h1, h2],
                     index=['type', 'group', 'PC', 'q25', 'q50', 'q75',
                            'h1', 'h2'])


transport = ['Aeolian', 'Fluvial', 'Glacial']
authors = ['this study', 'Smith_2018', 'Kalinska-Nartisa_2017', 'Sweet_2016',
           'Stevic_2015', 'Mahaney_1996']

# Run pca.fit_transform() on modern data for all-textures and mechanical
# ordinations
modern_aa, pca_aa = run_PCA_fit_transform(modern, tex_allauthors)
modern_me, pca_me = run_PCA_fit_transform(modern, tex_mechanical)

# Run pca.transform() on ancient data using pca object from pca.fit_transform()
ancient_aa = run_PCA_transform(ancient, tex_allauthors, pca_aa)
ancient_me = run_PCA_transform(ancient, tex_mechanical, pca_me)

# Group data into modern and ancient groups
data_modern = [modern_aa, modern_me]
data_ancient = [ancient_aa, ancient_me]

# Define empty "statistics" object to fill
statistics = pd.DataFrame(columns=['type', 'group', 'PC', 'q25', 'q50',
                                   'q75', 'h1', 'h2'],
                          index=np.arange(0, 18))

# Calculate Statistics for Aeolian, Fluvial, and Glacial Samples
for i in range(int(len(statistics)/3)):
    for j in range(3):
        if i*3+j < 9:
            if i == 0:
                statistics.loc[i*3+j, :] = calc_boxplot(modern_aa,
                                                        'All Textures',
                                                        'transport',
                                                        transport[j],'PC1')
            elif i == 1:
                statistics.loc[i*3+j, :] = calc_boxplot(modern_aa,
                                                        'All Textures',
                                                        'transport',
                                                        transport[j], 'PC2')
            elif i == 2:
                statistics.loc[i*3+j, :] = calc_boxplot(modern_aa,
                                                        'All Textures',
                                                        'transport',
                                                        transport[j], 'PC3')
        elif i*3+j >= 9:
            if i == 3:
                statistics.loc[i*3+j, :] = calc_boxplot(modern_me,
                                                        'Mechanical',
                                                        'transport',
                                                        transport[j],'PC1')
            elif i == 4:
                statistics.loc[i*3+j, :] = calc_boxplot(modern_me,
                                                        'Mechanical',
                                                        'transport',
                                                        transport[j], 'PC2')
            elif i == 5:
                statistics.loc[i*3+j, :] = calc_boxplot(modern_me,
                                                        'Mechanical',
                                                        'transport',
                                                        transport[j], 'PC3')
# Convert statistics object to excel sheet
statistics.to_excel('Data_XLSX/STATISTICS.xlsx')

# Calculate Statistics for Authors
statistics = pd.DataFrame(columns=['type', 'group', 'PC', 'q25', 'q50',
                                   'q75', 'h1', 'h2'],
                          index=np.arange(0, 36))
for i in range(int(len(statistics)/6)):
    for j in range(6):
        if i*6+j < 18:
            if i == 0:
                statistics.loc[i*6+j, :] = calc_boxplot(modern_aa,
                                                        'All Textures',
                                                        'author',
                                                        authors[j],'PC1')
            elif i == 1:
                statistics.loc[i*6+j, :] = calc_boxplot(modern_aa,
                                                        'All Textures',
                                                        'author',
                                                        authors[j], 'PC2')
            elif i == 2:
                statistics.loc[i*6+j, :] = calc_boxplot(modern_aa,
                                                        'All Textures',
                                                        'author',
                                                        authors[j], 'PC3')
        elif i*6+j >= 18:
            if i == 3:
                statistics.loc[i*6+j, :] = calc_boxplot(modern_me,
                                                        'Mechanical',
                                                        'author',
                                                        authors[j],'PC1')
            elif i == 4:
                statistics.loc[i*6+j, :] = calc_boxplot(modern_me,
                                                        'Mechanical',
                                                        'author',
                                                        authors[j], 'PC2')
            elif i == 5:
                statistics.loc[i*6+j, :] = calc_boxplot(modern_me,
                                                        'Mechanical',
                                                        'author',
                                                        authors[j], 'PC3')
# Convert statistics object to excel sheet
statistics.to_excel('Data_XLSX/STATISTICS-AUTHOR.xlsx')

# Sort ancient data by transport list below (e.g. Bravika first,
# aeolian second, etc.)
transport = ['Fluvial', 'Glacial', 'Aeolian', 'Bravika']
ancient_aa.transport = ancient_aa.transport.astype('category')
ancient_aa.transport.cat.set_categories(transport, inplace=True)
ancient_me.transport = ancient_me.transport.astype('category')
ancient_me.transport.cat.set_categories(transport, inplace=True)
ancient_aa = ancient_aa.sort_values(['transport'])
ancient_me = ancient_me.sort_values(['transport'])

# Plot Transport Boxplot (BOXPLOT-TRANSPORT)
fig, ax = plt.subplots(3, 2, figsize=(11, 15))
for i in range(3):
    for j in range(2):
        ax[i, j].tick_params(axis='both', direction='in', which='major',
                             top=True, labeltop=False, right=True,
                             labelright=False, left=True, bottom=True,
                             labelsize=20)
        if j == 0:
            sns.boxplot(x='transport', y='PC' + str(i+1), order=transport,
                        palette=['#0072B2', '#F0E442', '#D55E00', '#000000'],
                        data=modern_aa, ax=ax[i, j], saturation=1,
                        notch=False, bootstrap=10000)
            for k in range(len(ancient_aa)):
                ax[i, j].scatter(ancient_aa['transport'].iloc[k],
                                 ancient_aa['PC' + str(i+1)].iloc[k],
                                 facecolors=ancient_aa['color'].iloc[k],
                                 edgecolors='k',
                                 s=200,
                                 marker=ancient_aa['marker'].iloc[k])
            ax[i, j].set_xticks(np.arange(0, 4))
            ax[i, j].set_xticklabels(['Fluvial', 'Glacial', 'Aeolian',
                                      'Bråvika'])
            ax[i, j].add_patch(Rectangle((-1.165, -5), 0.33, 11, 
                                         clip_on=False, fill=True,
                                         facecolor='#648FFF', edgecolor='w'))
            ax[i, j].text(-1, 0.5, 'PC'+str(i+1), size=24, c='w',
                              horizontalalignment='center',
                              verticalalignment='center', weight='bold',
                              rotation=90)
            ax[i, j].text(-0.25, 5, 'A'+str(i+1), size=30,
                          horizontalalignment='center',
                          verticalalignment='center')
            if i == 0:
                mi, ma = ax[i, j].get_xlim()
                ax[i, j].add_patch(Rectangle((mi, 6.5),
                                          ma-mi,
                                          1, clip_on=False,
                                          fill=True, facecolor='#648FFF',
                                          edgecolor='w'))
                ax[i, j].text(mi + (ma-mi)/2, 7, 'All Textures', size=24, c='w',
                          horizontalalignment='center',
                          verticalalignment='center', weight='bold',
                          rotation=0)
        elif j == 1:
            sns.boxplot(x='transport', y='PC' + str(i+1), order=transport,
                        palette=['#0072B2', '#F0E442', '#D55E00', '#000000'],
                        data=modern_me, ax=ax[i, j], saturation=1,
                        notch=False, bootstrap=10000)
            for k in range(len(ancient_me)):
                ax[i, j].scatter(ancient_me['transport'].iloc[k],
                                 ancient_me['PC' + str(i+1)].iloc[k],
                                 facecolors=ancient_me['color'].iloc[k],
                                 edgecolors='k',
                                 s=200,
                                 marker=ancient_me['marker'].iloc[k])
            ax[i, j].set_xticks(np.arange(0, 4))
            ax[i, j].set_xticklabels(['Fluvial', 'Glacial', 'Aeolian',
                                      'Bråvika'])
            ax[i, j].text(-0.25, 5, 'B'+str(i+1), size=30,
                          horizontalalignment='center',
                          verticalalignment='center')
            if i == 0:
                mi, ma = ax[i, j].get_xlim()
                ax[i, j].add_patch(Rectangle((mi, 6.5),
                                          ma-mi,
                                          1, clip_on=False,
                                          fill=True, facecolor='#648FFF',
                                          edgecolor='w'))
                ax[i, j].text(mi + (ma-mi)/2, 7, 'Mechanical', size=24, c='w',
                          horizontalalignment='center',
                          verticalalignment='center', weight='bold',
                          rotation=0)
        ax[i, j].set_ylim(-5, 6)
        ax[i, j].set_ylabel('')
        ax[i, j].set_xlabel('')
plt.tight_layout()
plt.savefig('Figures/BOXPLOT-TRANSPORT.jpg', dpi=300)

# Plot Authors Boxplot (BOXPLOT-AUTHORS)
fig, ax = plt.subplots(3, 2, figsize=(10, 15))
for i in range(3):
    for j in range(2):
        ax[i, j].tick_params(axis='both', direction='in', which='major',
                             top=True, labeltop=False, right=True,
                             labelright=False, left=True, bottom=True,
                             labelsize=14)
        if j == 0:
            sns.boxplot(x='author', y='PC' + str(i+1),
                        order=['this study', 'Smith_2018',
                               'Kalinska-Nartisa_2017', 'Sweet_2016',
                               'Stevic_2015', 'Mahaney_1996'],
                        data=modern_aa, ax=ax[i, j], saturation=1,
                        notch=False, bootstrap=10000)
            ax[i, j].set_xticks(np.arange(0, 6))
            ax[i, j].set_xticklabels(['this study', 'Smith et al. (2018)',
                                      'Kalińska-Nartiša et al. (2017)',
                                      'Sweet and Brannan (2016)',
                                      'Stevic (2015)',
                                      'Mahaney et al. (1996)'], rotation=20,
                                     ha='right')
            ax[i, j].add_patch(Rectangle((-1.5, -5), 0.5, 11, 
                                         clip_on=False, fill=True,
                                         facecolor='#648FFF', edgecolor='w'))
            ax[i, j].text(-1.225, 0.5, 'PC'+str(i+1), size=18, c='w',
                              horizontalalignment='center',
                              verticalalignment='center', weight='bold',
                              rotation=90)
            ax[i, j].text(-0, 5, 'A'+str(i+1), size=30,
                          horizontalalignment='center',
                          verticalalignment='center')
            if i == 0:
                mi, ma = ax[i, j].get_xlim()
                ax[i, j].add_patch(Rectangle((mi, 6.5),
                                          ma-mi,
                                          1, clip_on=False,
                                          fill=True, facecolor='#648FFF',
                                          edgecolor='w'))
                ax[i, j].text(mi + (ma-mi)/2, 7, 'All Textures', size=18, c='w',
                          horizontalalignment='center',
                          verticalalignment='center', weight='bold')
        elif j == 1:
            sns.boxplot(x='author', y='PC' + str(i+1),
                        order=['this study', 'Smith_2018',
                               'Kalinska-Nartisa_2017', 'Sweet_2016',
                               'Stevic_2015', 'Mahaney_1996'],
                        data=modern_me, ax=ax[i, j], saturation=1,
                        notch=False, bootstrap=10000)
            ax[i, j].set_xticks(np.arange(0, 6))
            ax[i, j].set_xticklabels(['this study', 'Smith et al. (2018)',
                                      'Kalińska-Nartiša et al. (2017)',
                                      'Sweet and Brannan (2016)',
                                      'Stevic (2015)',
                                      'Mahaney et al. (1996)'], rotation=20,
                                     ha='right')
            ax[i, j].text(-0, 5, 'B'+str(i+1), size=30,
                          horizontalalignment='center',
                          verticalalignment='center')
            if i == 0:
                mi, ma = ax[i, j].get_xlim()
                ax[i, j].add_patch(Rectangle((mi, 6.5),
                                          ma-mi,
                                          1, clip_on=False,
                                          fill=True, facecolor='#648FFF',
                                          edgecolor='w'))
                ax[i, j].text(mi + (ma-mi)/2, 7, 'Mechanical', size=18, c='w',
                          horizontalalignment='center',
                          verticalalignment='center', weight='bold',
                          rotation=0)
        ax[i, j].set_ylim(-5, 6)
        ax[i, j].set_ylabel('')
        ax[i, j].set_xlabel('')
plt.tight_layout()
plt.savefig('Figures/BOXPLOT-AUTHOR.jpg', dpi=200)

plt.show()