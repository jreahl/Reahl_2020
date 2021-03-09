#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 14:30:25 2020

@author: jocelynreahl
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('Data_CSV/ALLDATA.csv')
tex_allpossible = ['af', 'as', 'bb', 'cf', 'ff', 'ls', 'saf', 'slf', 'up',  # Polygenetic
                    'er', 'vc',  # Percussion
                    'crg', 'cg', 'dt', 'sg',  # High-stress
                    'de', 'pf',  # Chemicals
                    'low', 'med', 'high']  # Relief
x = np.arange(len(tex_allpossible))
barwidth = 0.45
glacial = data.loc[4, 'af':'high'].reindex(index=tex_allpossible)
aeolian = data.loc[93, 'af':'high'].reindex(index=tex_allpossible)
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x-(0.5*barwidth), aeolian, barwidth, label='Aeolian', color='#D55E00',
       align='center')
ax.bar(x+(0.5*barwidth), glacial, barwidth, label='Glacial', color='#F0E442',
       align='center')
ax.legend()
ax.set_xticks(np.arange(len(tex_allpossible)))
ax.set_xticklabels(tex_allpossible)
ax.set_xlabel('Microtextures')
ax.set_ylabel('Probability of Occurrence')
plt.savefig('Figures/AGU-BAR.jpg', dpi=300)

fig, ax = plt.subplots(figsize=(6, 10))
ax.tick_params(axis='both', direction='in', which='major',
                             top=True, labeltop=False, right=True,
                             labelright=False, left=True, bottom=True,
                             labelsize=16)
ax.barh(x+(0.5*barwidth), aeolian.reindex(list(reversed(tex_allpossible))),
        barwidth, label='Aeolian', color='#D55E00', align='center')
ax.barh(x-(0.5*barwidth), glacial.reindex(list(reversed(tex_allpossible))),
        barwidth, label='Glacial', color='#F0E442', align='center')
ax.legend(fontsize=16)
ax.set_yticks(np.arange(len(tex_allpossible)))
ax.set_yticklabels(list(reversed(tex_allpossible)))
ax.set_ylabel('Microtextures', size=20)
ax.set_xlabel('Probability of Occurrence', size=20)
plt.tight_layout()
plt.savefig('Figures/AGU-BARH.jpg', dpi=300)

# OK and now for something completely different!
original = pd.read_excel('Data_XLSX/Culver_data.xlsx')
data = original.loc[:, 'A':'FF']
colors = ['C0', 'C0', 'C1', 'C1', 'C2', 'C2', 'C3', 'C3', 'C4', 'C4', 'C5',
          'C5', 'C6', 'C6', 'C7', 'C7']
fig, ax = plt.subplots()
for i in range(len(colors)):
    ax.plot(list(data.columns), data.iloc[i, :], color=colors[i])
fig, ax = plt.subplots()
cmap = plt.cm.get_cmap('plasma', 10)
img = plt.imshow(np.array([[0, 1]]), cmap=cmap)
img.set_visible(False)
plt.colorbar(orientation='vertical')
plt.show()