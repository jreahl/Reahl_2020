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


# Bringing in some functions that don't do any PCA stuff but just make
# things a bit easier to work with + calculate ellipses
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

# Import original dataset (master2 is an old naming schema, doesn't mean
# anything critical)
master2 = pd.read_csv('ALLDATA.csv')
# Define microtextures used to create PCA datasets
tex_allauthors = ['as', 'cf', 'cg', 'er', 'ls', 'pf', 'saf', 'slf', 'vc',
                  'low', 'med', 'high']
tex_mechanical = ['as', 'cf', 'cg', 'er', 'ls', 'saf', 'slf', 'vc', 'low',
                  'med', 'high']
# Divide original dataset into individual parts and apply RepackagedData class
modern = master2[master2['relage'] == 'Active']
modern = RepackagedData(modern, tex_mechanical)
ancient = master2[master2['transport'] == 'Bravika']
ancient = RepackagedData(ancient, tex_mechanical)
alldata = RepackagedData(master2, tex_mechanical)

# Spare code in the event I want to split the modern data in half to
# train + test

# np.random.seed(23)
# modern_reidx = modern.reindex(np.random.permutation(modern.index))
# trainingdata = modern_reidx.iloc[0:int(len(modern_reidx)/2), :]
# testdata = modern_reidx.iloc[int(len(modern_reidx)/2):int(len(modern_reidx)), :]
# trainingdata = RepackagedData(trainingdata, tex_mechanical)
# testdata = RepackagedData(testdata, tex_mechanical)

# Apply PCA to training data and define objects; create initial fit and
# transform data
pca = PCA() # define PCA object to use for the rest of this thing
scaled_data_train = preprocessing.scale(modern.data) # scale training data
pca_fit = pca.fit(scaled_data_train) # fit training data
pca_train = pca.transform(scaled_data_train) # transform training data
eigenvalues_train = pca.explained_variance_
per_var_train = np.round(pca.explained_variance_ratio_*100, decimals=2) # percent variance (per_var) of PC's using training data
components_train = ['PC' + str(x) for x in range(1, len(per_var_train)+1)] # just a list of PC's (PC1, PC2, PC3, etc.)
pca_df_train = pd.DataFrame(pca_train, columns=components_train) # make a dataframe out of the training data transformation to plot
# training data loading scores
loading_scores_train = pd.DataFrame(list(zip(tex_mechanical,
                                             pca.components_[0],
                                             pca.components_[1],
                                             pca.components_[2])),
                                    index=None,
                                    columns=['microtextures', 'PC1', 'PC2',
                                             'PC3'])

# Transform test data to existing PCA fit defined by training data
scaled_data_test = preprocessing.scale(ancient.data) # scale test data
pca_test = pca.transform(scaled_data_test) # transform test data using training data
# percent variance (per_var) of PC's used w/test data. These are the same as the training data because they are the training data's
eigenvalues_test = pca.explained_variance_
per_var_test = np.round(pca.explained_variance_ratio_*100, decimals=2) 
components_test = ['PC' + str(x) for x in range(1, len(per_var_test)+1)] # just another list of PC's
pca_df_test = pd.DataFrame(pca_test, columns=components_test) # make a dataframe w/test data

# Create a new PCA fit with all of the data and transform all of the data to the model
scaled_data_alldata = preprocessing.scale(alldata.data) # scale all data
pca_alldata = pca.fit_transform(scaled_data_alldata) # perform pca.fit() and pca.transform() on all the data; same as two lines of code
per_var_alldata = np.round(pca.explained_variance_ratio_*100, decimals=2) # percent variance of all data, there are different from before
eigenvalues_alldata = pca.explained_variance_
components_alldata = ['PC' + str(x) for x in range(1, len(per_var_alldata)+1)] # just a list of PC's
pca_df_alldata = pd.DataFrame(pca_alldata, columns=components_alldata) # make a dataframe to plot all data

# Plot all of the functions
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
for i in range(3):
    ax[i].set_xlabel('PC1')
    ax[i].set_ylabel('PC2')
    ax[i].set_xlim(-5, 5)
    ax[i].set_ylim(-5, 5)
# Create the first plot; Just training data:
# Plot the samples
for k in range(len(modern.data)):
    fk = modern.transportcolor[k]
    mk = modern.marker.iloc[k]
    xk = pca_df_train.PC1[k]
    yk = pca_df_train.PC2[k]
    ax[0].scatter(xk, yk, marker=mk, facecolors=fk, edgecolors='k', s=200,
                  alpha=1, linewidths=1)
# Create the ellipses
pca_df_train['color'] = modern.transportcolor
colors = ['#D55E00', '#0072B2', '#F0E442']
for c in colors:
    data = pca_df_train[pca_df_train['color'] == c]
    PC1mean = data['PC1'].mean()
    PC2mean = data['PC2'].mean()
    cov = np.cov(data['PC1'], data['PC2'])
    e = get_cov_ellipse(cov, (PC1mean, PC2mean), 1.96, facecolor='none',
                        edgecolor=c, alpha=1)
    ax[0].add_artist(e)

# Create the second plot; Training data + Test data
# Plot the samples of the training data
for k in range(len(modern.data)):
    fk = modern.transportcolor[k]
    mk = modern.marker.iloc[k]
    xk = pca_df_train.PC1[k]
    yk = pca_df_train.PC2[k]
    ax[1].scatter(xk, yk, marker=mk, facecolors=fk, edgecolors=fk, s=200,
                  alpha=0.5, linewidths=1)
# Create the ellipses for the training data
pca_df_train['color'] = modern.transportcolor
colors = ['#D55E00', '#0072B2', '#F0E442']
for c in colors:
    data = pca_df_train[pca_df_train['color'] == c]
    PC1mean = data['PC1'].mean()
    PC2mean = data['PC2'].mean()
    cov = np.cov(data['PC1'], data['PC2'])
    e = get_cov_ellipse(cov, (PC1mean, PC2mean), 1.96, facecolor='none',
                        edgecolor=c, alpha=1)
    ax[1].add_artist(e)
# Plot the samples of the test data
for k in range(len(ancient.data)):
    fk = ancient.transportcolor[k]
    mk = ancient.marker.iloc[k]
    xk = pca_df_test.PC1[k]
    yk = pca_df_test.PC2[k]
    ax[1].scatter(xk, yk, marker=mk, facecolors=fk, edgecolors='k', s=200,
                  alpha=1, linewidths=1)
# # Create the ellipses for the training data
# pca_df_test['color'] = ancient.transportcolor
# colors = ['#D55E00', '#0072B2', '#F0E442']
# for c in colors:
#     data = pca_df_test[pca_df_test['color'] == c]
#     PC1mean = data['PC1'].mean()
#     PC2mean = data['PC2'].mean()
#     cov = np.cov(data['PC1'], data['PC2'])
#     e = get_cov_ellipse(cov, (PC1mean, PC2mean), 1.96, facecolor='none',
#                         edgecolor=c, alpha=1)
#     ax[1].add_artist(e)

# Create the third plot; Just the alldata
# Plot all of the data
for k in range(len(alldata.data)):
    fk = alldata.transportcolor[k]
    mk = alldata.marker.iloc[k]
    xk = pca_df_alldata.PC1[k]
    yk = pca_df_alldata.PC2[k]
    ax[2].scatter(xk, yk, marker=mk, facecolors=fk, edgecolors='k', s=200,
               alpha=1, linewidths=1)
# Create the transport ellipses
pca_df_alldata['color'] = alldata.transportcolor
colors = ['#D55E00', '#0072B2', '#F0E442']
for c in colors:
    data = pca_df_alldata[pca_df_alldata['color'] == c]
    PC1mean = data['PC1'].mean()
    PC2mean = data['PC2'].mean()
    cov = np.cov(data['PC1'], data['PC2'])
    e = get_cov_ellipse(cov, (PC1mean, PC2mean), 1.96, facecolor='none',
                        edgecolor=c, alpha=1)
    ax[2].add_artist(e)

plt.savefig('Figures/PCA-TROUBLESHOOTING-3.jpg', dpi=300)
plt.show()

# Check if per_var for each value matches/doesn't match as expected:
print("Does the training data eigenvalues match the test data's?")
print('(Should be True)')
print(eigenvalues_train == eigenvalues_test)
print('')
print("Does the training data eigenvalues match that of all the data?")
print('(Should be False)')
print(eigenvalues_train == eigenvalues_alldata)
