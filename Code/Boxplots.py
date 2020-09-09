#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 15:34:15 2020

@author: jocelynreahl
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from sklearn import preprocessing
from sklearn.decomposition import PCA
from scipy import stats
import numpy as np

sns.set(style='white')
master2 = pd.read_csv('ALLDATA.csv')
tex_allauthors = ['as', 'cf', 'cg', 'er', 'ls', 'pf', 'saf', 'slf', 'vc',
                  'low', 'med', 'high']
tex_mechanical = ['as', 'cf', 'cg', 'er', 'ls', 'saf', 'slf', 'vc', 'low',
                  'med', 'high']
modern = master2[master2['relage'] == 'Active']
ancient = master2[master2['relage'] == 'Ancient']


def run_PCA_fit_transform(dataset, tex):
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
modern_aa, pca_aa = run_PCA_fit_transform(modern, tex_allauthors)
modern_me, pca_me = run_PCA_fit_transform(modern, tex_mechanical)
ancient_aa = run_PCA_transform(ancient, tex_allauthors, pca_aa)
ancient_me = run_PCA_transform(ancient, tex_mechanical, pca_me)
data_modern = [modern_aa, modern_me]
data_ancient = [ancient_aa, ancient_me]
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
statistics.to_excel('STATISTICS.xlsx')

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
statistics.to_excel('STATISTICS-AUTHOR.xlsx')

transport = ['Bravika', 'Aeolian', 'Fluvial', 'Glacial']
ancient_aa.transport = ancient_aa.transport.astype('category')
ancient_aa.transport.cat.set_categories(transport, inplace=True)
ancient_me.transport = ancient_me.transport.astype('category')
ancient_me.transport.cat.set_categories(transport, inplace=True)

ancient_aa = ancient_aa.sort_values(['transport'])
ancient_me = ancient_me.sort_values(['transport'])
fig, ax = plt.subplots(2, 3, figsize=(15, 10))
for i in range(2):
    for j in range(3):
        ax[i, j].tick_params(axis='both', which='major', top=True,
                             labeltop=False, right=True, labelright=False,
                             left=True, bottom=True, labelsize=14)
        if i == 0:
            # sns.stripplot(x='transport', y='PC' + str(j+1),
            #               order=['Bravika', 'Aeolian', 'Fluvial',
            #                       'Glacial'],
            #               palette=['#000000', '#D55E00', '#0072B2',
            #                         '#F0E442'], data=ancient_aa,
            #               ax=ax[i, j])
            sns.boxplot(x='transport', y='PC' + str(j+1), order=transport,
                        palette=['#000000', '#D55E00', '#0072B2', '#F0E442'],
                        data=modern_aa, ax=ax[i, j], saturation=1,
                        notch=False, bootstrap=10000)
            for k in range(len(ancient_aa)):
                ax[i, j].scatter(ancient_aa['transport'].iloc[k],
                                 ancient_aa['PC' + str(j+1)].iloc[k],
                                 facecolors=ancient_aa['color'].iloc[k],
                                 edgecolors='k',
                                 s=200,
                                 marker=ancient_aa['marker'].iloc[k])
            #     sns.stripplot(x='transport',
            #                   y='PC' + str(j+1),
            #                   order=['Bravika', 'Aeolian', 'Fluvial',
            #                           'Glacial'],
            #                   palette=['#000000', '#D55E00', '#0072B2',
            #                             '#F0E442'], data=ancient_aa,
            #                   ax=ax[i, j],
            #                   marker=ancient_aa['marker'].iloc[k])
            ax[i, j].set_xticks(np.arange(0, 4))
            ax[i, j].set_xticklabels(['Bråvika', 'Aeolian', 'Fluvial', 'Glacial'])
            mi, ma = ax[i, j].get_xlim()
            ax[i, j].add_patch(Rectangle((mi, 6.5),
                                          ma-mi,
                                          1, clip_on=False,
                                          fill=True, facecolor='#648FFF',
                                          edgecolor='w'))
            ax[i, j].text(mi + (ma-mi)/2, 7, 'PC'+str(j+1), size=18, c='w',
                          horizontalalignment='center',
                          verticalalignment='center', weight='bold',
                          rotation=0)
            if j == 0:
                ax[i, j].add_patch(Rectangle((-1.165, -5), 0.33, 11, 
                                         clip_on=False, fill=True,
                                         facecolor='#648FFF', edgecolor='w'))
                ax[i, j].text(-1, 1, 'All Textures', size=18, c='w',
                              horizontalalignment='center',
                              verticalalignment='center', weight='bold',
                              rotation=90)
                ax[i, j].text(0.1, 5, 'A1', size=30,
                              horizontalalignment='center',
                              verticalalignment='center')
            elif j == 1:
                ax[i, j].text(0.1, 5, 'A2', size=30,
                              horizontalalignment='center',
                              verticalalignment='center')
            elif j == 2:
                ax[i, j].text(0.1, 5, 'A3', size=30,
                              horizontalalignment='center',
                              verticalalignment='center')
        elif i == 1:
            sns.boxplot(x='transport', y='PC' + str(j+1), order=transport,
                        palette=['#000000', '#D55E00', '#0072B2', '#F0E442'],
                        data=modern_me, ax=ax[i, j], saturation=1,
                        notch=False, bootstrap=10000)
            for k in range(len(ancient_aa)):
                ax[i, j].scatter(ancient_me['transport'].iloc[k],
                                 ancient_me['PC' + str(j+1)].iloc[k],
                                 facecolors=ancient_me['color'].iloc[k],
                                 edgecolors='k',
                                 s=200,
                                 marker=ancient_me['marker'].iloc[k])
            ax[i, j].set_xticks(np.arange(0, 4))
            ax[i, j].set_xticklabels(['Bråvika', 'Aeolian', 'Fluvial', 'Glacial'])
            if j == 0:
                ax[i, j].add_patch(Rectangle((-1.165, -5), 0.33, 11, 
                                         clip_on=False, fill=True,
                                         facecolor='#648FFF', edgecolor='w'))
                ax[i, j].text(-1, 1, 'Mechanical', size=18, c='w',
                              horizontalalignment='center',
                              verticalalignment='center', weight='bold',
                              rotation=90)
                ax[i, j].text(0.1, 5, 'B1', size=30,
                              horizontalalignment='center',
                              verticalalignment='center')
            elif j == 1:
                ax[i, j].text(0.1, 5, 'B2', size=30,
                              horizontalalignment='center',
                              verticalalignment='center')
            elif j == 2:
                ax[i, j].text(0.1, 5, 'B3', size=30,
                              horizontalalignment='center',
                              verticalalignment='center')
        ax[i, j].set_ylim(-5, 6)
        ax[i, j].set_ylabel('')
        ax[i, j].set_xlabel('')

plt.tight_layout()
plt.savefig('Figures/BOXPLOT.jpg', dpi=300)

fig, ax = plt.subplots(2, 3, figsize=(15, 12))
for i in range(2):
    for j in range(3):
        ax[i, j].tick_params(axis='both', which='major', top=True,
                             labeltop=False, right=True, labelright=False,
                             left=True, bottom=True, labelsize=14)
        if i == 0:
            sns.boxplot(x='author', y='PC' + str(j+1), # hue='transport',
                        order=['this study', 'Smith_2018',
                               'Kalinska-Nartisa_2017', 'Sweet_2016',
                               'Stevic_2015', 'Mahaney_1996'],
                        # hue_order=['Aeolian', 'Fluvial', 'Glacial'],
                        # palette=['#D55E00', '#0072B2', '#F0E442'],
                        data=modern_aa, ax=ax[i, j], saturation=1)
            ax[i, j].add_patch(Rectangle((-0.5, 6.5), 6, 1, clip_on=False,
                                         fill=True, facecolor='#648FFF',
                                         edgecolor='w'))
            ax[i, j].text(2.5, 7, 'PC'+str(j+1), size=18, c='w',
                          horizontalalignment='center',
                          verticalalignment='center', weight='bold',
                          rotation=0)
            if j == 0:
                ax[i, j].add_patch(Rectangle((-1.5, -5), 0.5, 11, 
                                         clip_on=False, fill=True,
                                         facecolor='#648FFF', edgecolor='w'))
                ax[i, j].text(-1.25, 1, 'All Textures', size=18, c='w',
                              horizontalalignment='center',
                              verticalalignment='center', weight='bold',
                              rotation=90)
                ax[i, j].text(-0, 5, 'A1', size=30,
                              horizontalalignment='center',
                              verticalalignment='center')
            elif j == 1:
                ax[i, j].text(-0, 5, 'A2', size=30,
                              horizontalalignment='center',
                              verticalalignment='center')
            elif j == 2:
                ax[i, j].text(-0, 5, 'A3', size=30,
                              horizontalalignment='center',
                              verticalalignment='center')
        elif i == 1:
            sns.boxplot(x='author', y='PC' + str(j+1), # hue='transport',
                        order=['this study', 'Smith_2018',
                               'Kalinska-Nartisa_2017', 'Sweet_2016',
                               'Stevic_2015', 'Mahaney_1996'],
                        # hue_order=['Aeolian', 'Fluvial', 'Glacial'],
                        # palette=['#D55E00', '#0072B2', '#F0E442'],
                        data=modern_me, ax=ax[i, j], saturation=1)
            if j == 0:
                ax[i, j].add_patch(Rectangle((-1.5, -5), 0.5, 11, 
                                         clip_on=False, fill=True,
                                         facecolor='#648FFF', edgecolor='w'))
                ax[i, j].text(-1.25, 1, 'Mechanical', size=18, c='w',
                              horizontalalignment='center',
                              verticalalignment='center', weight='bold',
                              rotation=90)
                ax[i, j].text(-0, 5, 'B1', size=30,
                              horizontalalignment='center',
                              verticalalignment='center')
            elif j == 1:
                ax[i, j].text(-0, 5, 'B2', size=30,
                              horizontalalignment='center',
                              verticalalignment='center')
            elif j == 2:
                ax[i, j].text(-0, 5, 'B3', size=30,
                              horizontalalignment='center',
                              verticalalignment='center')
        ax[i, j].set_ylim(-5, 6)
        ax[i, j].set_ylabel('')
        ax[i, j].set_xlabel('')
        ax[i, j].set_xticks(np.arange(len(authors)))
        ax[i, j].set_xticklabels(authors, rotation=23, ha='right')

plt.tight_layout()
plt.savefig('Figures/BOXPLOT_AUTHOR.jpg', dpi=300)
plt.show()