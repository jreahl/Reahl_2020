#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 22:23:14 2020

@author: jocelynreahl
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Ellipse

master2 = pd.read_csv('ALLDATA.csv')
tex_allauthors = ['as', 'cf', 'cg', 'er', 'ls', 'pf', 'saf', 'slf', 'vc',
                  'low', 'med', 'high']
tex_mechanical = ['as', 'cf', 'cg', 'er', 'ls', 'saf', 'slf', 'vc', 'low',
                  'med', 'high']

class RepackagedData:
    def __init__(self, data, tex):
        self.data = data.loc[:, tex]
        self.transportcolor = list(map(lambda s: s.replace('\ufeff', ''),
                                   data.transportcolor))
        self.climatecolor = list(map(lambda s: s.replace('\ufeff', ''),
                                 data.climatecolor))
        self.marker = data.marker


def get_cov_ellipse(cov, center, nstd, **kwargs):
    """
    Derived from "BMI data with confidence ellipses" example in: Hill, C.,
    2016, Chapter 7: Matplotlib, in Hill, C., ed., Learning Scientific
    Programming in Python: Cambridge University Press.
    ---------------------------------------------------------------------------
    Draw an ellipse representing the confidence interval (in number of standard
    deviations (nstd))
    ---------------------------------------------------------------------------
    cov = covariance matrix of x and y values in plot
    center = averaged x and y values in plot; for centering the ellipse
    nstd = the number of standard deviations (95% confidence interval = 1.96)
    **kwargs = keyword arguments that would be used in the Ellipse function
    """
    # Find and sort eigenvalues and eigenvectors into descending order
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    # The anti-clockwise angle to rotate our ellipse by
    vx, vy = eigvecs[:, 0][0], eigvecs[:, 0][1]
    theta = np.arctan2(vy, vx)

    # Width and height of ellipse to draw
    width, height = 2 * nstd * np.sqrt(eigvals)
    return Ellipse(xy=center, width=width, height=height,
                   angle=np.degrees(theta), lw=2, **kwargs)


modern = master2[master2['relage'] == 'Active']
ancient = master2[master2['transport'] == 'Bravika']
ancient = RepackagedData(ancient, tex_mechanical)
np.random.seed(23)
modern_reidx = modern.reindex(np.random.permutation(modern.index))
trainingdata = modern_reidx.iloc[0:int(len(modern_reidx)/2), :]
testdata = modern_reidx.iloc[int(len(modern_reidx)/2):int(len(modern_reidx)), :]
trainingdata = RepackagedData(trainingdata, tex_mechanical)
testdata = RepackagedData(testdata, tex_mechanical)

scaled_data_train = preprocessing.scale(trainingdata.data)
pca = PCA()
pca_train = pca.fit_transform(scaled_data_train)
per_var_train = np.round(pca.explained_variance_ratio_*100, decimals=2)
components_train = ['PC' + str(x) for x in range(1, len(per_var_train)+1)]
pca_df_train = pd.DataFrame(pca_train, columns=components_train)
loading_scores_train = pd.DataFrame(list(zip(tex_mechanical,
                                             pca.components_[0],
                                             pca.components_[1],
                                             pca.components_[2])),
                                    index=None,
                                    columns=['microtextures', 'PC1', 'PC2',
                                             'PC3'])

scaled_data_test = preprocessing.scale(testdata.data)
pca_test = pca.transform(scaled_data_test)
per_var_test = np.round(pca.explained_variance_ratio_*100, decimals=2)
components_test = ['PC' + str(x) for x in range(1, len(per_var_test)+1)]
pca_df_test = pd.DataFrame(pca_test, columns=components_test)

scaled_data_ancient = preprocessing.scale(ancient.data)
pca_ancient = pca.transform(scaled_data_ancient)
per_var_ancient = np.round(pca.explained_variance_ratio_*100, decimals=2)
components_ancient = ['PC' + str(x) for x in range(1, len(per_var_ancient)+1)]
pca_df_ancient = pd.DataFrame(pca_ancient, columns=components_ancient)

fig, ax = plt.subplots(figsize=(10, 10))
for k in range(len(trainingdata.data)):
    fk = trainingdata.transportcolor[k]
    mk = trainingdata.marker.iloc[k]
    xk = pca_df_train.PC1[k]
    yk = pca_df_train.PC2[k]
    ax.scatter(xk, yk, marker=mk, facecolors=fk, edgecolors='k', s=200,
               alpha=1, linewidths=1)
pca_df_train['color'] = trainingdata.transportcolor
colors = ['#D55E00', '#0072B2', '#F0E442']
for c in colors:
    data = pca_df_train[pca_df_train['color'] == c]
    PC1mean = data['PC1'].mean()
    PC3mean = data['PC3'].mean()
    cov = np.cov(data['PC1'], data['PC3'])
    e = get_cov_ellipse(cov, (PC1mean, PC3mean), 1.96,
                                            facecolor='none', edgecolor=c,
                                            alpha=1)
    ax.add_artist(e)
for k in range(len(testdata.data)):
    fk = testdata.transportcolor[k]
    mk = testdata.marker.iloc[k]
    xk = pca_df_test.PC1[k]
    yk = pca_df_test.PC2[k]
    ax.scatter(xk, yk, marker=mk, facecolors=fk, edgecolors='k', s=200,
               alpha=1, linewidths=1)
pca_df_test['color'] = testdata.transportcolor
for c in colors:
    data = pca_df_test[pca_df_test['color'] == c]
    PC1mean = data['PC1'].mean()
    PC3mean = data['PC3'].mean()
    cov = np.cov(data['PC1'], data['PC3'])
    e = get_cov_ellipse(cov, (PC1mean, PC3mean), 1.96,
                                            facecolor='none', edgecolor=c,
                                            alpha=1)
    ax.add_artist(e)
for k in range(len(ancient.data)):
    fk = ancient.transportcolor[k]
    mk = ancient.marker.iloc[k]
    xk = pca_df_ancient.PC1[k]
    yk = pca_df_ancient.PC2[k]
    ax.scatter(xk, yk, marker=mk, facecolors=fk, edgecolors='k', s=200,
               alpha=1, linewidths=1)
plt.show()
