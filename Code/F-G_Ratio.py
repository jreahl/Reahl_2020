#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 17:22:24 2020

@author: jocelynreahl
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

master2 = pd.read_csv('ALLDATA.csv')

polygenetic = ['af', 'as', 'bb', 'cf', 'ff', 'ls', 'saf', 'slf', 'up']
percussion = ['er', 'vc']
histress = ['crg', 'cg', 'dt', 'sg']
group = [polygenetic, percussion, histress]
total = polygenetic + percussion + histress

sweet = master2[master2['author'] == 'Sweet_2016']
smith_loc = ['Norway', 'California', 'Puerto-Rico', 'Peru']
smith = [master2[(master2['author'] == 'Smith_2018') &
                 (master2['sample'].str.contains(loc))] for loc in smith_loc]
legend_elements = [Line2D([0], [0], color='C0', label='Sweet and Brannan (2016)'),
                   Line2D([0], [0], color='C1', label='Norway - Smith et al. (2018)'),
                   Line2D([0], [0], color='C2', label='California - Smith et al. (2018)'),
                   Line2D([0], [0], color='C3', label='Puerto Rico - Smith et al. (2018)'),
                   Line2D([0], [0], color='C4', label='Peru - Smith et al. (2018)')]


def relative(data):
    polygenetic = ['af', 'as', 'bb', 'cf', 'ff', 'ls', 'saf', 'slf', 'up']
    fluvial = ['er', 'vc']
    glacial = ['crg', 'cg', 'dt', 'sg']
    group = [polygenetic, fluvial, glacial]
    total = polygenetic + fluvial + glacial
    apices = pd.DataFrame(0, index=data.index, columns=['Polygenetic',
                                                        'Percussion',
                                                        'High-Stress',
                                                        'F/G'])
    for i in list(apices.index):
        for j in range(len(group)):
            valuesum = 0
            denominator = 0
            for val in group[j]:
                if np.isnan(data.loc[list(apices.index)[0], val]) == True:
                    group[j].remove(val)
                    total.remove(val)
                else:
                    valuesum += data.loc[i, val]
            for t in total:
                denominator += data.loc[i, t]
            apices.loc[i, list(apices.columns)[j]] = (valuesum/denominator)*100
            apices.loc[i, 'F/G'] = apices.loc[i, 'Percussion']/apices.loc[i, 'High-Stress']
    return apices


def plt_scatter_bestfit(ax, x, y, color):
    ax.scatter(x, y, color=color)
    m, b = np.polyfit(x, y, 1)
    ax.plot(x, m*x + b, color=color)
    correlation_matrix = np.corrcoef(x, y)
    r = correlation_matrix[0, 1]
    r2 = r**2
    ax.text(x.iloc[-1], m*x.iloc[-1]+b, 'R\u00b2 = '+'{:.2f}'.format(r2), color=color)


sweet_relative = relative(sweet)
smith_relative = [relative(smith[i]) for i in range(len(smith_loc))]
sweet_rolling = sweet_relative.loc[:, 'F/G'].rolling(window=5, center=True).mean()
smith_rolling = [smith_relative[i].loc[:, 'F/G'].rolling(window=5, center=True).mean()
                 for i in range(len(smith_loc))]

fig, ax = plt.subplots(4, 1, figsize=(10, 15))
ax[0].set_xscale('log')
ax[0].set_xlim(10**0, 10**2.5)
# ax[0].set_xlabel('Transport Distance [km]')
ax[0].set_ylabel('Polygenetic [%]')
plt_scatter_bestfit(ax[0], sweet['trans-distance'], sweet_relative['Polygenetic'], 'C0')
for i in range(len(smith_relative)):
    plt_scatter_bestfit(ax[0], smith[i]['trans-distance'], smith_relative[i]['Polygenetic'], 'C'+str(i+1))
ax[0].legend(handles=legend_elements)

ax[1].set_xscale('log')
ax[1].set_xlim(10**0, 10**2.5)
# ax[1].set_xlabel('Transport Distance [km]')
ax[1].set_ylabel('High-Stress [%]')
plt_scatter_bestfit(ax[1], sweet['trans-distance'], sweet_relative['High-Stress'], 'C0')
for i in range(len(smith_relative)):
    plt_scatter_bestfit(ax[1], smith[i]['trans-distance'], smith_relative[i]['High-Stress'], 'C'+str(i+1))

ax[2].set_xscale('log')
ax[2].set_xlim(10**0, 10**2.5)
# ax[2].set_xlabel('Transport Distance [km]')
ax[2].set_ylabel('Percussion [%]')
plt_scatter_bestfit(ax[2], sweet['trans-distance'], sweet_relative['Percussion'], 'C0')
for i in range(len(smith_relative)):
    plt_scatter_bestfit(ax[2], smith[i]['trans-distance'], smith_relative[i]['Percussion'], 'C'+str(i+1))

ax[3].set_xscale('log')
ax[3].set_xlim(10**0, 10**2.5)
ax[3].set_xlabel('Transport Distance [km]')
ax[3].set_ylabel('F/G Ratio')
plt_scatter_bestfit(ax[3], sweet['trans-distance'], sweet_relative['F/G'], 'C0')
for i in range(len(smith_relative)):
    plt_scatter_bestfit(ax[3], smith[i]['trans-distance'], smith_relative[i]['F/G'], 'C'+str(i+1))
plt.savefig('Figures/Transport-Comparison.jpg', dpi=300)
plt.show()
# Need to do the kind of math I did for the ternary diagram in here.
# Also need to do this with a separate spreadsheet where I include transport
# distance as a column.
# Or honestly do I?? Maybe the only things I change are the JIRP samples
# and the glacial/fluvial samples from Sweet and Brannan; thinking about
# fluvially-dominated versus glacially-dominated samples being the reasoning
# for samples ending up in fluvial vs. glacial bins; wouldn't apply to other
# studies that focus on just discussing fluvial samples as they define them.