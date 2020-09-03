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


def run_PCA(dataset, tex):
    data = dataset.loc[:, tex]
    scaled_data = preprocessing.scale(data)
    pca = PCA()
    pca_ref = pca.fit_transform(scaled_data)
    per_var = np.round(pca.explained_variance_ratio_*100,
                                    decimals=2)
    components = ['PC' + str(x) for x in range(1, len(per_var)+1)]
    pca_df = pd.DataFrame(pca_ref, columns=components)
    pca_df['transport'] = dataset['transport']
    return pca_df


def calc_boxplot(dataset, ordination, transport, PC):
    d = dataset[dataset['transport'] == transport].loc[:, PC]
    q25, q50, q75 = np.percentile(d, [25, 50, 75])
    whiskerlim = 1.5 * stats.iqr(d)
    h1 = np.min(d[d >= (q25 - whiskerlim)])
    h2 = np.max(d[d <= (q75 + whiskerlim)])
    return pd.Series([ordination, transport, PC, q25, q50, q75, h1, h2],
                     index=['type', 'transport', 'PC', 'q25', 'q50', 'q75',
                            'h1', 'h2'])


transport = ['Aeolian', 'Fluvial', 'Glacial']
allauthors = run_PCA(modern, tex_allauthors)
mechanical = run_PCA(modern, tex_mechanical)
data = [allauthors, mechanical]
statistics = pd.DataFrame(columns=['type', 'transport', 'PC', 'q25', 'q50',
                                   'q75', 'h1', 'h2'],
                          index=np.arange(0, 18))

for i in range(int(len(statistics)/3)):
    for j in range(3):
        if i*3+j < 9:
            if i == 0:
                statistics.loc[i*3+j, :] = calc_boxplot(allauthors,
                                                        'alltextures',
                                                        transport[j],'PC1')
            elif i == 1:
                statistics.loc[i*3+j, :] = calc_boxplot(allauthors,
                                                        'alltextures',
                                                        transport[j], 'PC2')
            elif i == 2:
                statistics.loc[i*3+j, :] = calc_boxplot(allauthors,
                                                        'alltextures',
                                                        transport[j], 'PC3')
        elif i*3+j >= 9:
            if i == 3:
                statistics.loc[i*3+j, :] = calc_boxplot(mechanical,
                                                        'mechanical',
                                                        transport[j],'PC1')
            elif i == 4:
                statistics.loc[i*3+j, :] = calc_boxplot(mechanical,
                                                        'mechanical',
                                                        transport[j], 'PC2')
            elif i == 5:
                statistics.loc[i*3+j, :] = calc_boxplot(mechanical,
                                                        'mechanical',
                                                        transport[j], 'PC3')
statistics.to_excel('STATISTICS.xlsx')

fig, ax = plt.subplots(2, 3, figsize=(15, 10))
for i in range(2):
    for j in range(3):
        ax[i, j].tick_params(axis='both', which='major', top=True,
                             labeltop=False, right=True, labelright=False,
                             left=True, bottom=True, labelsize=14)
        if i == 0:
            sns.boxplot(x='transport', y='PC' + str(j+1), order=transport,
                        palette=['#D55E00', '#0072B2', '#F0E442'],
                        data=allauthors, ax=ax[i, j], saturation=1)
            ax[i, j].add_patch(Rectangle((-0.5, 6.5), 3, 1, clip_on=False,
                                         fill=True, facecolor='#648FFF',
                                         edgecolor='w'))
            ax[i, j].text(1, 7, 'PC'+str(j+1), size=18, c='w',
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
                ax[i, j].text(-0.165, 5, 'A1', size=30,
                              horizontalalignment='center',
                              verticalalignment='center')
            elif j == 1:
                ax[i, j].text(-0.165, 5, 'A2', size=30,
                              horizontalalignment='center',
                              verticalalignment='center')
            elif j == 2:
                ax[i, j].text(-0.165, 5, 'A3', size=30,
                              horizontalalignment='center',
                              verticalalignment='center')
        elif i == 1:
            sns.boxplot(x='transport', y='PC' + str(j+1), order=transport,
                        palette=['#D55E00', '#0072B2', '#F0E442'],
                        data=mechanical, ax=ax[i, j], saturation=1)
            if j == 0:
                ax[i, j].add_patch(Rectangle((-1.165, -5), 0.33, 11, 
                                         clip_on=False, fill=True,
                                         facecolor='#648FFF', edgecolor='w'))
                ax[i, j].text(-1, 1, 'Mechanical', size=18, c='w',
                              horizontalalignment='center',
                              verticalalignment='center', weight='bold',
                              rotation=90)
                ax[i, j].text(-0.165, 5, 'B1', size=30,
                              horizontalalignment='center',
                              verticalalignment='center')
            elif j == 1:
                ax[i, j].text(-0.165, 5, 'B2', size=30,
                              horizontalalignment='center',
                              verticalalignment='center')
            elif j == 2:
                ax[i, j].text(-0.165, 5, 'B3', size=30,
                              horizontalalignment='center',
                              verticalalignment='center')
        ax[i, j].set_ylim(-5, 6)
        ax[i, j].set_ylabel('')
        ax[i, j].set_xlabel('')

plt.tight_layout()
plt.savefig('Figures/BOXPLOT.jpg', dpi=300)
plt.show()