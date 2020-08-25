#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 11:38:44 2020

@author: Jocelyn N. Reahl
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as col
from matplotlib.patches import Rectangle
import seaborn as sns


def heatmap(sampleage):
    '''
    A function to create heatmaps for the Modern and Ancient data.
    ----------
    sampleage : str
        A string to identify the set of data to plot.
        'MODERN' = plot the modern dataset
        'ANCIENT' = plot the ancient dataset
        Ideally input these as all caps strings, but there's an if statement
        to convert lowercase inputs to uppercase.

    Returns
    -------
    None.

    '''
    sns.set() # Initialize Seaborn
    
    # Check if sampleage string is capitalized:
    if sampleage.isupper() == False: # if input isn't in all caps, make it all caps
        sampleage = sampleage.upper()
    else: # otherwise move along
        pass
    
    # Import original data + get it ready for plotting in a heatmap:
    original = pd.read_csv('HEATMAP-' + sampleage + '.csv') # Original dataset
    original = original.set_index('sample') # Reindex "original" using sample names
    if sampleage == 'MODERN':
        transmodes = ['Aeolian', 'Fluvial', 'Glacial'] # Transport mode groups
        transcolors = ['#D55E00', '#0072B2', '#F0E442'] # Trans. mode colors
    elif sampleage == 'ANCIENT':
        transmodes = ['Unknown', 'Aeolian', 'Fluvial', 'Glacial'] # Trans. modes
        transcolors = ['#000000', '#D55E00', '#0072B2', '#F0E442'] # Colors
    microtextures = ['af', 'as', 'bb', 'cf', 'ff', 'ls', 'saf', 'slf', # Polygenetic
                     'er', 'vc',  # Percussion
                     'crg', 'cg', 'dt', 'sg', 'up', # High-stress
                     'de', 'pf',  # Chemicals
                     'low', 'med', 'high']  # Relief
    
    # Separate samples by transport mode and reindex columns to order in
    # "microtextures" object above:
    groupedsamples = [original[original['transport'] == t] for t in transmodes]
    data = [g.loc[:, 'af':'high'] for g in groupedsamples]
    data = [d.reindex(columns=microtextures) for d in data]
    
    # Make list of colormaps from white (0) to color in transcolors (1):
    cmaps = [col.LinearSegmentedColormap.from_list("", ['w', c]) for c in transcolors]
    
    # Plot the heatmap (subplots used dependent on 'MODERN' vs. 'ANCIENT'):
    if sampleage == 'MODERN':
        fig, ax = plt.subplots(3, 1, figsize=(15, 13),
                               gridspec_kw={'height_ratios':[1, 0.67, 2.11]})
    elif sampleage == 'ANCIENT':
        fig, ax = plt.subplots(4, 1, figsize=(15, 11),
                               gridspec_kw={'height_ratios':[1, 1.5, 3, 8]})
    fig.subplots_adjust(hspace=0.05) # Reduce spacing between subplots
    
    for d, c, a, co, t in zip(data, cmaps, ax, transcolors, transmodes):
        # Plot the data to the hundreths decimal place:
        sns.heatmap(d, annot=True, cmap=c, cbar=False, ax=a, vmin=0, vmax=1,
                    yticklabels=False, fmt='.2f')
        
        # Manually annotate y-labels (would use yticklabels=True in
        # sns.heatmap() but labels plot weirdly for certain heatmap subplot
        # lengths):
        for s in range(len(list(d.index))):
            if 'AVG' not in d.index[s]:
                a.text(-0.25, s+0.5, d.index[s], horizontalalignment='right',
                       verticalalignment='center', fontsize=11)
            elif 'AVG' in d.index[s]:
                a.text(-0.25, s+0.5, d.index[s], horizontalalignment='right',
                       verticalalignment='center', fontsize=14,
                       fontweight='bold')
            a.set_ylabel('')
            
            # Create white outline boxes w/no fill to section off polygenetic,
            # percussion, high-stress, chemical, and relief microtextures on
            # each subplot.
            a.add_patch(Rectangle((0, 0), 8, len(d), fill=False,
                                  edgecolor='w'))
            a.add_patch(Rectangle((8, 0), 2, len(d), fill=False,
                                  edgecolor='w'))
            a.add_patch(Rectangle((10, 0), 5, len(d), fill=False,
                                  edgecolor='w'))
            a.add_patch(Rectangle((15, 0), 2, len(d), fill=False,
                                  edgecolor='w'))
            a.add_patch(Rectangle((17, 0), 3, len(d), fill=False,
                                  edgecolor='w'))
            
            # Create transport mode labels to right of plot; "Unknown" gets
            # a shortened version of "UNK" because "Unknown" can't fit in the
            # length offered by the two Bråvika Mbr. samples.
            if t != 'Unknown':
                a.text(20, len(d)/2, t, rotation=270, size=20,
                       verticalalignment='center')
            elif t == 'Unknown':
                a.text(20, len(d)/2, 'UNK', rotation=270, size=20,
                       verticalalignment='center')
            
            # Add patches + text to distinguish the polygenetic, percussion,
            # high-stress, chemical, and relief microtextures @ top of plot.
            if a == ax[0]:
                # Patches:
                a.add_patch(Rectangle((0, -1-0.3), 8, 1, clip_on=False,
                                      fill=True, facecolor='#648FFF',
                                      edgecolor='w'))
                a.add_patch(Rectangle((8, -1-0.3), 2, 1, clip_on=False,
                                      fill=True, facecolor='#785EF0',
                                      edgecolor='w'))
                a.add_patch(Rectangle((10, -1-0.3), 5, 1, clip_on=False,
                                      fill=True, facecolor='#DC267F',
                                      edgecolor='w'))
                a.add_patch(Rectangle((15, -1-0.3), 2, 1, clip_on=False,
                                      fill=True, facecolor='#FE6100',
                                      edgecolor='w'))
                a.add_patch(Rectangle((17, -1-0.3), 3, 1, clip_on=False,
                                      fill=True, facecolor='#FFB000',
                                      edgecolor='w'))
                # Text:
                a.text(4, -0.5-0.3, 'Polygenetic', horizontalalignment='center',
                       verticalalignment='center', fontsize=14, c='w',
                       weight='bold')
                a.text(9, -0.5-0.3, 'Percussion', horizontalalignment='center',
                       verticalalignment='center', fontsize=14, c='w',
                       weight='bold')
                a.text(12.5, -0.5-0.3, 'High-Stress', horizontalalignment='center',
                       verticalalignment='center', fontsize=14, c='w',
                       weight='bold')
                a.text(16, -0.5-0.3, 'Chemical', horizontalalignment='center',
                       verticalalignment='center', fontsize=14, c='w',
                       weight='bold')
                a.text(18.5, -0.5-0.3, 'Relief', horizontalalignment='center',
                       verticalalignment='center', fontsize=14, c='w',
                       weight='bold')
                a.set_xticklabels('')
            
            # Create x label @ bottom of last subplot; otherwise no xticklabels
            elif a == ax[-1]:
                a.set_xlabel('Microtextures', size=20)
            else:
                a.set_xticklabels('')
    plt.tight_layout() # Reduce overall white space in figure
    plt.savefig('Figures/HEATMAP-' + sampleage + '.jpg', dpi=300) # Save!
    plt.show() # Show plot in console


heatmap('MODERN')
heatmap('ANCIENT')
