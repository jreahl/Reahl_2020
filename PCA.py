#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Jocelyn N. Reahl
Title: PCA
Description: Script to generate PCA plots for all-textures and mechanical
ordinations, as well as blank legend templates to further edit in a vector
graphics program like Illustrator or Inkscape.

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
alldata = pd.read_csv('Data_CSV/ALLDATA.csv')

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

# Set up the custom legend for this PCA plot
# - Blank spaces for microtextures need to be replaced with abbreviations for
#   each microtexture in a vector graphics program like Illustrator or
#   Inkscape; had trouble getting it to do it in the actual script so this is
#   the hack for now.
# - If using groupby='transport' in the PCAplot function, comment out the 'Ice
#   Nearby/Absent' + blank space and uncomment out the transport modes
#   ('Aeolian', 'Fluvial', 'Glacial', + blank space).
# - If using groupby='climate' in the PCAplot function, uncomment the 'Ice
#   Nearby/Absent' + blank space and comment out the transport modes
#   ('Aeolian', 'Fluvial', 'Glacial', + blank space).
# - If doing an all-textures ordination, uncomment the precipitated features
#   line and blank space; if doing mechanical leave it commented.
legend_elements = [
                   Patch(facecolor='#ffffff', label=''),
                   Patch(facecolor='#D55E00', label='Aeolian'),
                   Patch(facecolor='#0072B2', label='Fluvial'),
                   Patch(facecolor='#F0E442', label='Glacial'),
                   # Patch(facecolor='#000000', label='Unknown (Bråvika Mbr.)'),
                   # Patch(facecolor='#ffffff', label=''),
                   # Patch(facecolor='#1A85FF',
                   #       label='Ice Nearby'),
                   # Patch(facecolor='#D41159',
                   #       label='Ice Absent'),
                   # Patch(facecolor='#000000',
                   #       label='Ancient'),
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
                   Patch(facecolor='#ffffff', label='High Relief')
                   ]
legend = plt.legend(handles=legend_elements, frameon=True)
plt.axis('off') # keep the matplotlib axis out of the legend

def export_legend(legend, filename='PCA-LEGEND.jpg', expand=[-5, -5, 5, 5],
                  dpi=300):
    '''
    Export legend as an image.
    ----------
    legend = legend object created using plt.legend()
    filename = str; chosen name of the document file + extension; file is
               directed to the Figures folder.
    expand = list; parameters to help create bbox to save fig.
    dpi = int; the dots per inch (dpi) of the new file; 300 is on the higher
          end and 72 is on the lower end, for reference.
    '''
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig('Figures/' + filename, dpi=dpi, bbox_inches=bbox)


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


def score_sorter(pca, tex, n):
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



def PCAplot(dataframe, tex, label, groupby='transport'):
    '''
    Perform PCA and plot biplots of PCA data and loadings.
    ----------
    dataframe = pandas Dataframe; use data from ALLDATA.csv
    tex = list of microtexture abbreviations to use for PCA ordination
    label = str; file name with no extension; directs to Figures folder
    groupby = str; 'transport' = samples are grouped by transport mode,
              'climate' = samples are grouped by ice presence/absence.
    '''
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
    # Print out Variance for PC1-PC3 in the console:
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
    ls = 24
    for i in range(3):
        for j in range(3):
            ax[i, j].tick_params(axis='both', direction='in', which='major',
                             top=True, labeltop=False, right=True,
                             labelright=False, left=True, bottom=True,
                             labelsize=ls)
            if j == 0:
                ax[i, j].set_xlim(-5, 6)
                ax[i, j].set_ylim(-5, 6)
                if i == 0:
                    ax[i, j].set_xlabel('PC1', size=ls)
                    ax[i, j].set_ylabel('PC2', size=ls)
                    ax[i, j].text(-4.5, 4.5, 'A1', size=40)
                    ax[i, j].add_patch(Rectangle((-5, 6+0.5), 11, 1,
                                                 clip_on=False, fill=True,
                                                 facecolor='#648FFF',
                                                 edgecolor='w'))
                    ax[i, j].text(0.5, 7, 'Modern Samples', size=ls, c='w',
                                  horizontalalignment='center',
                                  verticalalignment='center', weight='bold')
                    ax[i, j].add_patch(Rectangle((-5-3.1, -5), 1, 11,
                                                 clip_on=False, fill=True,
                                                 facecolor='#648FFF',
                                                 edgecolor='w'))
                    ax[i, j].text(-7.5, 0.5, 'PC1 v. PC2', size=ls, c='w',
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
                        confidence_ellipse(data['PC1'], data['PC2'],
                                           ax[i, j], n_std=2, facecolor='none',
                                           edgecolor=c, alpha=1, lw=2)
                elif i == 1:
                    ax[i, j].set_xlabel('PC1', size=ls)
                    ax[i, j].set_ylabel('PC3', size=ls)
                    ax[i, j].text(-4.5, 4.5, 'B1', size=40)
                    ax[i, j].add_patch(Rectangle((-5-3.1, -5), 1, 11,
                                                 clip_on=False, fill=True,
                                                 facecolor='#648FFF',
                                                 edgecolor='w'))
                    ax[i, j].text(-7.5, 0.5, 'PC1 v. PC3', size=ls, c='w',
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
                        confidence_ellipse(data['PC1'], data['PC3'],
                                           ax[i, j], n_std=2, facecolor='none',
                                           edgecolor=c, alpha=1, lw=2)
                elif i == 2:
                    ax[i, j].set_xlabel('PC2', size=ls)
                    ax[i, j].set_ylabel('PC3', size=ls)
                    ax[i, j].text(-4.5, 4.5, 'C1', size=40)
                    ax[i, j].add_patch(Rectangle((-5-3.1, -5), 1, 11,
                                                 clip_on=False, fill=True,
                                                 facecolor='#648FFF',
                                                 edgecolor='w'))
                    ax[i, j].text(-7.5, 0.5, 'PC2 v. PC3', size=ls, c='w',
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
                        confidence_ellipse(data['PC2'], data['PC3'],
                                           ax[i, j], n_std=2, facecolor='none',
                                           edgecolor=c, alpha=1, lw=2)
            elif j == 1:
                ax[i, j].set_xlim(-0.6, 0.6)
                ax[i, j].set_ylim(-0.7, 0.7)
                if i == 0:
                    ax[i, j].set_xlabel('PC1', size=ls)
                    # ax[i, j].set_ylabel('PC2', size=ls)
                    ax[i, j].text(-0.545, 0.51, 'A2', size=40)
                    ax[i, j].add_patch(Rectangle((-0.6, 0.7+0.064), 1.2, 0.127,
                                                 clip_on=False, fill=True,
                                                 facecolor='#648FFF',
                                                 edgecolor='w'))
                    ax[i, j].text(0, 0.7+0.064+(0.127/2), 'PC Loadings',
                                  size=ls, c='w',
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
                    ax[i, j].set_xlabel('PC1', size=ls)
                    # ax[i, j].set_ylabel('PC3', size=ls)
                    ax[i, j].text(-0.545, 0.51, 'B2', size=40)
                    for labels, PC1, PC3 in zip(loading_scores_ref['microtextures'],
                                                loading_scores_ref['PC1'],
                                                loading_scores_ref['PC3']):
                        ax[i, j].arrow(0, 0, PC1, PC3, color='#000000',
                                 length_includes_head=True, head_width=0.02,
                                 overhang=0.05)
                        textplot(ax[i, j], tex, PC1, PC3, 0, 2, labels)
                elif i == 2:
                    ax[i, j].set_xlabel('PC2', size=ls)
                    # ax[i, j].set_ylabel('PC3', size=ls)
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
                    ax[i, j].set_xlabel('PC1', size=ls)
                    # ax[i, j].set_ylabel('PC2', size=ls)
                    ax[i, j].text(-4.5, 4.5, 'A3', size=40)
                    ax[i, j].add_patch(Rectangle((-5, 6+0.5), 11, 1,
                                                 clip_on=False, fill=True,
                                                 facecolor='#648FFF',
                                                 edgecolor='w'))
                    ax[i, j].text(0.5, 7, 'Ancient Samples', size=ls, c='w',
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
                        confidence_ellipse(data['PC1'], data['PC2'],
                                           ax[i, j], n_std=2, facecolor='none',
                                           edgecolor=c, alpha=1, lw=2)
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
                    ax[i, j].set_xlabel('PC1', size=ls)
                    # ax[i, j].set_ylabel('PC3', size=ls)
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
                        confidence_ellipse(data['PC1'], data['PC3'],
                                           ax[i, j], n_std=2, facecolor='none',
                                           edgecolor=c, alpha=1, lw=2)
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
                    ax[i, j].set_xlabel('PC2', size=ls)
                    # ax[i, j].set_ylabel('PC3', size=ls)
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
                        confidence_ellipse(data['PC2'], data['PC3'],
                                           ax[i, j], n_std=2, facecolor='none',
                                           edgecolor=c, alpha=1, lw=2)
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
    plt.savefig('Figures/PCA-' + label.upper() + '.jpg', dpi=300)
    plt.show()
