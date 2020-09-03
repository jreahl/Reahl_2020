#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Ellipse
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.legend_handler import HandlerBase
from matplotlib.legend import Legend

master2 = pd.read_csv('ALLDATA.csv')
tex_allpossible = ['af', 'as', 'bb', 'cf', 'ff', 'ls', 'saf', 'slf', 'up',  # Polygenetic
                   'er', 'vc',  # Percussion
                   'crg', 'cg', 'dt', 'sg',  # High-stress
                   'de', 'pf',  # Chemicals
                   'low', 'med', 'high']  # General (applies to all grains)
tex_allauthors = ['as', 'cf', 'cg', 'er', 'ls', 'pf', 'saf', 'slf', 'vc',
                  'low', 'med', 'high']
tex_mechanical = ['as', 'cf', 'cg', 'er', 'ls', 'saf', 'slf', 'vc', 'low',
                  'med', 'high']


legend_elements = [
                   Patch(facecolor='#ffffff', label=''),
                   Patch(facecolor='#D55E00', label='Aeolian'),
                   Patch(facecolor='#0072B2', label='Fluvial'),
                   Patch(facecolor='#F0E442', label='Glacial'),
                   Patch(facecolor='#000000', label='Unknown (Bråvika Mbr.)'),
                   # Patch(facecolor='#ffffff', label=''),
                   # Patch(facecolor='#1A85FF',
                   #       label='Ice Nearby'),
                   # Patch(facecolor='#D41159',
                   #       label='Ice Absent'),
                   Patch(facecolor='#ffffff', label=''),
                   Patch(facecolor='#ffffff', label=''),
                   Line2D([0], [0], marker='o', color='w',
                          markerfacecolor='w', markeredgecolor='k', ms=10,
                          label='This Study'),
                   Line2D([0], [0], marker='s', color='w',
                          markerfacecolor='w', markeredgecolor='k', ms=10,
                          label='Smith et al. (2018)'),
                   Line2D([0], [0], marker='p', color='w',
                          markerfacecolor='w', markeredgecolor='k', ms=10,
                          label='Kalińska-Nartiša et al. (2017)'),
                   Line2D([0], [0], marker='*', color='w',
                          markerfacecolor='w', markeredgecolor='k', ms=10,
                          label='Sweet and Brannan (2016)'),
                   Line2D([0], [0], marker='H', color='w',
                          markerfacecolor='w', markeredgecolor='k', ms=10,
                          label='Stevic (2015)'),
                   Line2D([0], [0], marker='^', color='w',
                          markerfacecolor='w', markeredgecolor='k', ms=10,
                          label='Mahaney et al. (1996)'),
                   Line2D([0], [0], marker='P', color='w',
                          markerfacecolor='w', markeredgecolor='k', ms=10,
                          label='Nartišs and Kalińska-Nartiša\n(2017)'),
                   Line2D([0], [0], marker='D', color='w',
                          markerfacecolor='w', markeredgecolor='k', ms=10,
                          label='Deane (2010)'),
                   Line2D([0], [0], marker='X', color='w',
                          markerfacecolor='w', markeredgecolor='k', ms=10,
                          label='Sweet and Soreghan (2010)'),
                   Line2D([0], [0], marker='d', color='w',
                          markerfacecolor='w', markeredgecolor='k', ms=10,
                          label='Mahaney et al. (2001)'),
                   Line2D([0], [0], marker='v', color='w',
                          markerfacecolor='w', markeredgecolor='k', ms=10,
                          label='Mahaney and Kalm (1995)'),
                   Patch(facecolor='#ffffff', label=''),
                   Patch(facecolor='#ffffff', label=''),
                   Patch(facecolor='#ffffff', label=''),
                   Patch(facecolor='#ffffff', label='Arc-Shaped Steps'),
                   Patch(facecolor='#ffffff', label='Conchoidal Fractures'),
                   Patch(facecolor='#ffffff', label='Linear Steps'),
                   Patch(facecolor='#ffffff', label='Sharp Angular Features'),
                   Patch(facecolor='#ffffff',
                         label='Subparallel Linear Fractures'),
                   Patch(facecolor='#ffffff', label=''),
                   Patch(facecolor='#ffffff', label='Edge Rounding'),
                   Patch(facecolor='#ffffff',
                         label='V-Shaped Percussion Cracks'),
                   Patch(facecolor='#ffffff', label=''),
                   Patch(facecolor='#ffffff', label='Curved Grooves'),
                    # Patch(facecolor='#ffffff', label=''),
                    # Patch(facecolor='#ffffff', label='Precipitated Features'),
                   Patch(facecolor='#ffffff', label=''),
                   Patch(facecolor='#ffffff', label='Low Relief'),
                   Patch(facecolor='#ffffff', label='Medium Relief'),
                   Patch(facecolor='#ffffff', label='High Relief')]
legend = plt.legend(handles=legend_elements, frameon=True)
plt.axis('off')

def export_legend(legend, filename='PCA-LEGEND.jpg', expand=[-5, -5, 5, 5], dpi=300):
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig('Figures/' + filename, dpi=dpi, bbox_inches=bbox)


class RepackagedData:
    def __init__(self, data, tex):
        self.data = data.loc[:, tex]
        self.transportcolor = list(map(lambda s: s.replace('\ufeff', ''),
                                   data.transportcolor))
        self.climatecolor = list(map(lambda s: s.replace('\ufeff', ''),
                                 data.climatecolor))
        self.marker = data.marker


def generate_trainingdata(dataframe, tex):
    transport = ['Aeolian', 'Fluvial', 'Glacial']
    sansBravika = dataframe[(dataframe['transport'] != 'Bravika') &
                            (dataframe['relage'] == 'Active')]
    initialdata = pd.DataFrame(columns=sansBravika.columns)
    sel_id = []
    if len(dataframe) % 2 == 0:
        maxsize = int(len(sansBravika)/2)
    elif len(dataframe) % 2 == 1:
        maxsize = int((len(sansBravika)+1)/2)
    for i in range(len(transport)):
        transport_data = sansBravika[sansBravika['transport'] == transport[i]]
        rand_id = random.choice(transport_data.index.values)
        r = transport_data.loc[rand_id, :]
        initialdata = initialdata.append(r)
        sel_id.append(rand_id)
    while len(initialdata) < maxsize:
        rand_id = random.choice(sansBravika.index.values)
        if rand_id not in sel_id:
            r = sansBravika.loc[rand_id, :]
            initialdata = initialdata.append(r)
            sel_id.append(rand_id)
        else:
            pass
    remainingdata = pd.DataFrame(columns=sansBravika.columns)
    remainder = len(sansBravika) - maxsize
    while len(remainingdata) < remainder:
        rand_id = random.choice(sansBravika.index.values)
        if rand_id not in sel_id:
            r = sansBravika.loc[rand_id, :]
            remainingdata = remainingdata.append(r)
            sel_id.append(rand_id)
        else:
            pass

    initialdata = RepackagedData(initialdata, tex)
    remainingdata = RepackagedData(remainingdata, tex)
    return initialdata, remainingdata


def textplot(ax, tex, x, y, PCx, PCy, label):
    """
    Function to plot text on PCA/NMDS plots in a pretty way.
    ---------------------------------------------------------------------------
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


def score_sorter(pca, tex, n):
    PCloadings = pd.Series(pca.components_[n-1], index=tex)
    PCsorted = PCloadings.sort_values(ascending=False)
    # PCsorted = PCsorted.index.values
    return PCsorted


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


def PCAplot_randomtrainingdata(dataframe, tex):
    sample = dataframe[dataframe['transport'] == 'Bravika Mbr']
    sample = RepackagedData(sample, tex)
    initialdata, remainingdata = generate_trainingdata(dataframe, tex)
    scaled_data = preprocessing.scale(initialdata.data)
    pca = PCA()
    pca_initialdata = pca.fit_transform(scaled_data)
    per_var = np.round(np.round(pca.explained_variance_ratio_*100, decimals=1))
    components = ['PC' + str(x) for x in range(1, len(per_var)+1)]
    pca_df = pd.DataFrame(pca_initialdata, columns=components)
    loading_scores = pd.DataFrame(list(zip(tex, pca.components_[0],
                                           pca.components_[1])), index=None,
                                  columns=['microtextures', 'PC1', 'PC2'])
    fig, ax = plt.subplots(figsize=(13, 10))
    ax.set_title('Trained Dataset Using Randomly Selected Training Data')
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_xlabel('PC1', size=20)
    ax.set_ylabel('PC2', size=20)
    ax.tick_params(axis='both', which='major', labelsize=14)
    for i in range(len(initialdata.data)):
        fi = initialdata.transportcolor[i]
        mi = initialdata.marker.iloc[i]
        xi = pca_df.PC1[i]
        yi = pca_df.PC2[i]
        ax.scatter(xi, yi, marker=mi, facecolors=fi, edgecolors='#000000',
                   s=200, alpha=1)
    for labels, PC1, PC2 in loading_scores.values[:, 0:3]:
        ax.arrow(0, 0, PC1*10, PC2*10, color='#989898',
                 length_includes_head=True, head_width=0.3, overhang=0.5)
        textplot(ax, PC1*10, PC2*10, labels)
    scaled_data_remain = preprocessing.scale(remainingdata.data)
    pca_remainingdata = pca.transform(scaled_data_remain)
    per_var_remain = np.round(np.round(pca.explained_variance_ratio_*100,
                                       decimals=1))
    components_remain = ['PC' + str(x) for x in range(1,
                                                      len(per_var_remain)+1)]
    pca_df_remain = pd.DataFrame(pca_remainingdata, columns=components_remain)
    for i in range(len(remainingdata.data)):
        fi = remainingdata.transportcolor[i]
        mi = remainingdata.marker.iloc[i]
        xi = pca_df_remain.PC1[i]
        yi = pca_df_remain.PC2[i]
        ax.scatter(xi, yi, marker=mi, facecolors=fi, edgecolors='#000000',
                   s=200, alpha=1)
    scaled_data_sam = preprocessing.scale(sample.data)
    pca_sample = pca.transform(scaled_data_sam)
    per_var_sam = np.round(np.round(pca.explained_variance_ratio_*100,
                                    decimals=1))
    components_sam = ['PC' + str(x) for x in range(1, len(per_var_sam)+1)]
    pca_df_sam = pd.DataFrame(pca_sample, columns=components_sam)
    for i in range(len(sample.data)):
        fi = sample.transportcolor[i]
        mi = sample.marker.iloc[i]
        xi = pca_df_sam.PC1[i]
        yi = pca_df_sam.PC2[i]
        ax.scatter(xi, yi, marker=mi, facecolors=fi, edgecolors='#000000',
                   s=200, alpha=1)
    ax.legend(handles=legend_elements, loc='upper left',
              bbox_to_anchor=(1, 1), ncol=1, prop={'size': 12})
    plt.tight_layout()
    plt.savefig('PCA_TrainedDataset.jpg', dpi=300)
    plt.show()



def PCAplot(dataframe, tex, groupby, label):
    # Define "reference" and "sample" datasets and repackage into
    # RepackagedData class.
    reference = dataframe[(dataframe['relage'] == 'Active')]
    reference = RepackagedData(reference, tex)
    if 'pf' in set(tex):
        sample = dataframe[(dataframe['relage'] != 'Active') &
                           (dataframe['author'] != 'Sweet_2010')]
    else:
        sample = dataframe[(dataframe['relage'] != 'Active')]
    sample = RepackagedData(sample, tex)
    
    # Perform PCA analysis on reference data
    scaled_data_ref = preprocessing.scale(reference.data)
    pca = PCA()
    pca_ref = pca.fit_transform(scaled_data_ref)
    per_var_ref = np.round(pca.explained_variance_ratio_*100,
                                    decimals=2)
    components_ref = ['PC' + str(x) for x in range(1, len(per_var_ref)+1)]
    pca_df_ref = pd.DataFrame(pca_ref, columns=components_ref)
    loading_scores_ref = pd.DataFrame(list(zip(tex,
                                               pca.components_[0],
                                               pca.components_[1],
                                               pca.components_[2])),
                                      index=None,
                                      columns=['microtextures', 'PC1', 'PC2',
                                               'PC3'])
    print('Principal Component Variance:')
    print(per_var_ref)
    print('Sum of PC1 = ' + str(per_var_ref[0]))
    print('Sum of PC1 + PC2 = ' + str(per_var_ref[0] + per_var_ref[1]))
    print('Sum of PC1 + PC2 + PC3 = ' + str(per_var_ref[0] +
                                            per_var_ref[1] +
                                            per_var_ref[2]))
    print('Microtextural Loading Scores:')
    print(loading_scores_ref)
    print('Sorted Scores:')
    PC1sorted = score_sorter(pca, tex, 1)
    PC2sorted = score_sorter(pca, tex, 2)
    PC3sorted = score_sorter(pca, tex, 3)
    print('PC1:')
    print(PC1sorted)
    print('PC2:')
    print(PC2sorted)
    print('PC3:')
    print(PC3sorted)
    
    # Transform sample data into PCA space set by reference data
    scaled_data_sam = preprocessing.scale(sample.data)
    pca_sam = pca.transform(scaled_data_sam)
    per_var_sam = np.round(np.round(pca.explained_variance_ratio_*100,
                                    decimals=1))
    components_sam = ['PC' + str(x) for x in range(1, len(per_var_sam)+1)]
    pca_df_sam = pd.DataFrame(pca_sam, columns=components_sam)
    
    # Generate a Scree Plot for the PCA reference plot
    plt.bar(x=range(1, len(per_var_ref)+1), height=per_var_ref,
            tick_label=components_ref)
    plt.ylabel('Percentage of Explained Variance [%]')
    plt.xlabel('Principal Component')
    
    # Plot PCA ordination w/reference first, then sample
    fig, ax = plt.subplots(3, 3, figsize=(20, 20)) # Set up axes
    for i in range(3):
        for j in range(3):
            ax[i, j].tick_params(axis='both', which='major', top=True,
                                 labeltop=False, right=True, labelright=False,
                                 labelsize=14)
            if j == 0:
                ax[i, j].set_xlim(-5, 6)
                ax[i, j].set_ylim(-5, 6)
                if i == 0:
                    ax[i, j].set_xlabel('PC1', size=20)
                    ax[i, j].set_ylabel('PC2', size=20)
                    ax[i, j].text(-4.5, 4.5, 'A1', size=40)
                    ax[i, j].add_patch(Rectangle((-5, 6+0.5), 11, 1,
                                                 clip_on=False, fill=True,
                                                 facecolor='#648FFF',
                                                 edgecolor='w'))
                    ax[i, j].text(0.5, 7, 'Modern Samples', size=20, c='w',
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
                    for k in range(len(reference.data)):
                        if groupby == 'transport':
                            fk = reference.transportcolor[k]
                        elif groupby == 'climate':
                            fk = reference.climatecolor[k]
                        mk = reference.marker.iloc[k]
                        xk = pca_df_ref.PC1[k]
                        yk = pca_df_ref.PC2[k]
                        ax[i, j].scatter(xk, yk, marker=mk, facecolors=fk,
                                         edgecolors='k', s=200, alpha=1,
                                         linewidths=1)
                    if groupby == 'transport':
                        pca_df_ref['color'] = reference.transportcolor
                        colors = ['#D55E00', '#0072B2', '#F0E442']
                    elif groupby == 'climate':
                        pca_df_ref['color'] = reference.climatecolor
                        colors = ['#1A85FF', '#D41159']
                    for c in colors:
                        data = pca_df_ref[pca_df_ref['color'] == c]
                        PC1mean = data['PC1'].mean()
                        PC2mean = data['PC2'].mean()
                        cov = np.cov(data['PC1'], data['PC2'])
                        e = get_cov_ellipse(cov, (PC1mean, PC2mean), 1.96,
                                            facecolor='none', edgecolor=c,
                                            alpha=1)
                        ax[i, j].add_artist(e)
                    # for k in range(len(sample.data)):
                    #     if color == 'transport':
                    #         fk = sample.transportcolor[k]
                    #     elif color == 'climate':
                    #         fk = sample.climatecolor[k]
                    #     mk = sample.marker.iloc[k]
                    #     xk = pca_df_sam.PC1[k]
                    #     yk = pca_df_sam.PC2[k]
                    #     ax[i, j].scatter(xk, yk, marker=mk, facecolors=fk,
                    #                      edgecolors='k', s=200, alpha=1,
                    #                      linewidths=1)
                elif i == 1:
                    ax[i, j].set_xlabel('PC1', size=20)
                    ax[i, j].set_ylabel('PC3', size=20)
                    ax[i, j].text(-4.5, 4.5, 'B1', size=40)
                    ax[i, j].add_patch(Rectangle((-5-3, -5), 1, 11,
                                                 clip_on=False, fill=True,
                                                 facecolor='#648FFF',
                                                 edgecolor='w'))
                    ax[i, j].text(-7.5, 0.5, 'PC1 v. PC3', size=20, c='w',
                                  horizontalalignment='center',
                                  verticalalignment='center', weight='bold',
                                  rotation=90)
                    for k in range(len(reference.data)):
                        if groupby == 'transport':
                            fk = reference.transportcolor[k]
                        elif groupby == 'climate':
                            fk = reference.climatecolor[k]
                        mk = reference.marker.iloc[k]
                        xk = pca_df_ref.PC1[k]
                        yk = pca_df_ref.PC3[k]
                        ax[i, j].scatter(xk, yk, marker=mk, facecolors=fk,
                                         edgecolors='k', s=200, alpha=1,
                                         linewidths=1)
                    if groupby == 'transport':
                        pca_df_ref['color'] = reference.transportcolor
                        colors = ['#D55E00', '#0072B2', '#F0E442']
                    elif groupby == 'climate':
                        pca_df_ref['color'] = reference.climatecolor
                        colors = ['#1A85FF', '#D41159']
                    for c in colors:
                        data = pca_df_ref[pca_df_ref['color'] == c]
                        PC1mean = data['PC1'].mean()
                        PC3mean = data['PC3'].mean()
                        cov = np.cov(data['PC1'], data['PC3'])
                        e = get_cov_ellipse(cov, (PC1mean, PC3mean), 1.96,
                                            facecolor='none', edgecolor=c,
                                            alpha=1)
                        ax[i, j].add_artist(e)
                    # for k in range(len(sample.data)):
                    #     if color == 'transport':
                    #         fk = sample.transportcolor[k]
                    #     elif color == 'climate':
                    #         fk = sample.climatecolor[k]
                    #     mk = sample.marker.iloc[k]
                    #     xk = pca_df_sam.PC1[k]
                    #     yk = pca_df_sam.PC3[k]
                    #     ax[i, j].scatter(xk, yk, marker=mk, facecolors=fk,
                    #                       edgecolors='k', s=200, alpha=1,
                    #                       linewidths=1)
                elif i == 2:
                    ax[i, j].set_xlabel('PC2', size=20)
                    ax[i, j].set_ylabel('PC3', size=20)
                    ax[i, j].text(-4.5, 4.5, 'C1', size=40)
                    ax[i, j].add_patch(Rectangle((-5-3, -5), 1, 11,
                                                 clip_on=False, fill=True,
                                                 facecolor='#648FFF',
                                                 edgecolor='w'))
                    ax[i, j].text(-7.5, 0.5, 'PC2 v. PC3', size=20, c='w',
                                  horizontalalignment='center',
                                  verticalalignment='center', weight='bold',
                                  rotation=90)
                    for k in range(len(reference.data)):
                        if groupby == 'transport':
                            fk = reference.transportcolor[k]
                        elif groupby == 'climate':
                            fk = reference.climatecolor[k]
                        mk = reference.marker.iloc[k]
                        xk = pca_df_ref.PC2[k]
                        yk = pca_df_ref.PC3[k]
                        ax[i, j].scatter(xk, yk, marker=mk, facecolors=fk,
                                         edgecolors='k', s=200, alpha=1,
                                         linewidths=1)
                    if groupby == 'transport':
                        pca_df_ref['color'] = reference.transportcolor
                        colors = ['#D55E00', '#0072B2', '#F0E442']
                    elif groupby == 'climate':
                        pca_df_ref['color'] = reference.climatecolor
                        colors = ['#1A85FF', '#D41159']
                    for c in colors:
                        data = pca_df_ref[pca_df_ref['color'] == c]
                        PC2mean = data['PC2'].mean()
                        PC3mean = data['PC3'].mean()
                        cov = np.cov(data['PC2'], data['PC3'])
                        e = get_cov_ellipse(cov, (PC2mean, PC3mean), 1.96,
                                            facecolor='none', edgecolor=c,
                                            alpha=1)
                        ax[i, j].add_artist(e)
                    # for k in range(len(sample.data)):
                    #     if color == 'transport':
                    #         fk = sample.transportcolor[k]
                    #     elif color == 'climate':
                    #         fk = sample.climatecolor[k]
                    #     mk = sample.marker.iloc[k]
                    #     xk = pca_df_sam.PC2[k]
                    #     yk = pca_df_sam.PC3[k]
                    #     ax[i, j].scatter(xk, yk, marker=mk, facecolors=fk,
                    #                      edgecolors='k', s=200, alpha=1,
                    #                      linewidths=1)
            elif j == 1:
                ax[i, j].set_xlim(-0.6, 0.6)
                ax[i, j].set_ylim(-0.7, 0.7)
                if i == 0:
                    ax[i, j].set_xlabel('PC1', size=20)
                    # ax[i, j].set_ylabel('PC2', size=20)
                    ax[i, j].text(-0.545, 0.51, 'A2', size=40)
                    ax[i, j].add_patch(Rectangle((-0.6, 0.7+0.064), 1.2, 0.127,
                                                 clip_on=False, fill=True,
                                                 facecolor='#648FFF',
                                                 edgecolor='w'))
                    ax[i, j].text(0, 0.7+0.064+(0.127/2), 'PC Loadings', size=20, c='w',
                                  horizontalalignment='center',
                                  verticalalignment='center', weight='bold')
                    for labels, PC1, PC2 in zip(loading_scores_ref['microtextures'],
                                                loading_scores_ref['PC1'],
                                                loading_scores_ref['PC2']):
                        ax[i, j].arrow(0, 0, PC1, PC2, color='#000000',
                                       length_includes_head=True,
                                       head_width=0.02, overhang=0.05)
                        textplot(ax[i, j], tex, PC1, PC2, 0, 1, labels)
                elif i == 1:
                    ax[i, j].set_xlabel('PC1', size=20)
                    # ax[i, j].set_ylabel('PC3', size=20)
                    ax[i, j].text(-0.545, 0.51, 'B2', size=40)
                    for labels, PC1, PC3 in zip(loading_scores_ref['microtextures'],
                                                loading_scores_ref['PC1'],
                                                loading_scores_ref['PC3']):
                        ax[i, j].arrow(0, 0, PC1, PC3, color='#000000',
                                 length_includes_head=True, head_width=0.02,
                                 overhang=0.05)
                        textplot(ax[i, j], tex, PC1, PC3, 0, 2, labels)
                elif i == 2:
                    ax[i, j].set_xlabel('PC2', size=20)
                    # ax[i, j].set_ylabel('PC3', size=20)
                    ax[i, j].text(-0.545, 0.51, 'C2', size=40)
                    for labels, PC2, PC3 in zip(loading_scores_ref['microtextures'],
                                                loading_scores_ref['PC2'],
                                                loading_scores_ref['PC3']):
                        ax[i, j].arrow(0, 0, PC2, PC3, color='#000000',
                                 length_includes_head=True, head_width=0.02,
                                 overhang=0.05)
                        textplot(ax[i, j], tex, PC2, PC3, 1, 2, labels)
            elif j == 2:
                ax[i, j].set_xlim(-5, 6)
                ax[i, j].set_ylim(-5, 6)
                if i == 0:
                    ax[i, j].set_xlabel('PC1', size=20)
                    ax[i, j].set_ylabel('PC2', size=20)
                    ax[i, j].text(-4.5, 4.5, 'A3', size=40)
                    ax[i, j].add_patch(Rectangle((-5, 6+0.5), 11, 1,
                                                 clip_on=False, fill=True,
                                                 facecolor='#648FFF',
                                                 edgecolor='w'))
                    ax[i, j].text(0.5, 7, 'Ancient Samples', size=20, c='w',
                                  horizontalalignment='center',
                                  verticalalignment='center', weight='bold')
                    for k in range(len(reference.data)):
                        if groupby == 'transport':
                            fk = reference.transportcolor[k]
                        elif groupby == 'climate':
                            fk = reference.climatecolor[k]
                        mk = reference.marker.iloc[k]
                        xk = pca_df_ref.PC1[k]
                        yk = pca_df_ref.PC2[k]
                        ax[i, j].scatter(xk, yk, marker=mk, facecolors=fk,
                                         edgecolors=fk, s=200, alpha=0.5,
                                         linewidths=1)
                    if groupby == 'transport':
                        pca_df_ref['color'] = reference.transportcolor
                        colors = ['#D55E00', '#0072B2', '#F0E442']
                    elif groupby == 'climate':
                        pca_df_ref['color'] = reference.climatecolor
                        colors = ['#1A85FF', '#D41159']
                    for c in colors:
                        data = pca_df_ref[pca_df_ref['color'] == c]
                        PC1mean = data['PC1'].mean()
                        PC2mean = data['PC2'].mean()
                        cov = np.cov(data['PC1'], data['PC2'])
                        e = get_cov_ellipse(cov, (PC1mean, PC2mean), 1.96,
                                            facecolor='none', edgecolor=c,
                                            alpha=1)
                        ax[i, j].add_artist(e)
                    for k in range(len(sample.data)):
                        if groupby == 'transport':
                            fk = sample.transportcolor[k]
                        elif groupby == 'climate':
                            fk = sample.climatecolor[k]
                        mk = sample.marker.iloc[k]
                        xk = pca_df_sam.PC1[k]
                        yk = pca_df_sam.PC2[k]
                        ax[i, j].scatter(xk, yk, marker=mk, facecolors=fk,
                                         edgecolors='k', s=200, alpha=1,
                                         linewidths=1)
                elif i == 1:
                    ax[i, j].set_xlabel('PC1', size=20)
                    # ax[i, j].set_ylabel('PC3', size=20)
                    ax[i, j].text(-4.5, 4.5, 'B3', size=40)
                    for k in range(len(reference.data)):
                        if groupby == 'transport':
                            fk = reference.transportcolor[k]
                        elif groupby == 'climate':
                            fk = reference.climatecolor[k]
                        mk = reference.marker.iloc[k]
                        xk = pca_df_ref.PC1[k]
                        yk = pca_df_ref.PC3[k]
                        ax[i, j].scatter(xk, yk, marker=mk, facecolors=fk,
                                         edgecolors=fk, s=200, alpha=0.5,
                                         linewidths=1)
                    if groupby == 'transport':
                        pca_df_ref['color'] = reference.transportcolor
                        colors = ['#D55E00', '#0072B2', '#F0E442']
                    elif groupby == 'climate':
                        pca_df_ref['color'] = reference.climatecolor
                        colors = ['#1A85FF', '#D41159']
                    for c in colors:
                        data = pca_df_ref[pca_df_ref['color'] == c]
                        PC1mean = data['PC1'].mean()
                        PC3mean = data['PC3'].mean()
                        cov = np.cov(data['PC1'], data['PC3'])
                        e = get_cov_ellipse(cov, (PC1mean, PC3mean), 1.96,
                                            facecolor='none', edgecolor=c,
                                            alpha=1)
                        ax[i, j].add_artist(e)
                    for k in range(len(sample.data)):
                        if groupby == 'transport':
                            fk = sample.transportcolor[k]
                        elif groupby == 'climate':
                            fk = sample.climatecolor[k]
                        mk = sample.marker.iloc[k]
                        xk = pca_df_sam.PC1[k]
                        yk = pca_df_sam.PC3[k]
                        ax[i, j].scatter(xk, yk, marker=mk, facecolors=fk,
                                          edgecolors='k', s=200, alpha=1,
                                          linewidths=1)
                elif i == 2:
                    ax[i, j].set_xlabel('PC2', size=20)
                    # ax[i, j].set_ylabel('PC3', size=20)
                    ax[i, j].text(-4.5, 4.5, 'C3', size=40)
                    for k in range(len(reference.data)):
                        if groupby == 'transport':
                            fk = reference.transportcolor[k]
                        elif groupby == 'climate':
                            fk = reference.climatecolor[k]
                        mk = reference.marker.iloc[k]
                        xk = pca_df_ref.PC2[k]
                        yk = pca_df_ref.PC3[k]
                        ax[i, j].scatter(xk, yk, marker=mk, facecolors=fk,
                                         edgecolors=fk, s=200, alpha=0.5,
                                         linewidths=1)
                    if groupby == 'transport':
                        pca_df_ref['color'] = reference.transportcolor
                        colors = ['#D55E00', '#0072B2', '#F0E442']
                    elif groupby == 'climate':
                        pca_df_ref['color'] = reference.climatecolor
                        colors = ['#1A85FF', '#D41159']
                    for c in colors:
                        data = pca_df_ref[pca_df_ref['color'] == c]
                        PC2mean = data['PC2'].mean()
                        PC3mean = data['PC3'].mean()
                        cov = np.cov(data['PC2'], data['PC3'])
                        e = get_cov_ellipse(cov, (PC2mean, PC3mean), 1.96,
                                            facecolor='none', edgecolor=c,
                                            alpha=1)
                        ax[i, j].add_artist(e)
                    for k in range(len(sample.data)):
                        if groupby == 'transport':
                            fk = sample.transportcolor[k]
                        elif groupby == 'climate':
                            fk = sample.climatecolor[k]
                        mk = sample.marker.iloc[k]
                        xk = pca_df_sam.PC2[k]
                        yk = pca_df_sam.PC3[k]
                        ax[i, j].scatter(xk, yk, marker=mk, facecolors=fk,
                                         edgecolors='k', s=200, alpha=1,
                                         linewidths=1) 
    # plt.tight_layout()
    plt.savefig('Figures/PCA-' + label.upper() + '.jpg', dpi=300)
    plt.show()
