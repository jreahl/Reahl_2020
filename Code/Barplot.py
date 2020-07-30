#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 14:35:13 2020

@author: jocelynreahl
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

master2 = pd.read_csv('ALLDATA.csv')
data = master2.loc[:, 'af':'high']
microtextures = ['high', 'med', 'low',
                 'pf', 'de',
                 'up', 'sg', 'dt', 'cg', 'crg',
                 'vc', 'er',
                 'slf', 'saf', 'ls', 'ff', 'cf', 'bb', 'as', 'af']
data = data.reindex(columns=microtextures)
Vanda = data.loc[0, :]
fig, ax = plt.subplots(figsize=(3, 5))
ax.barh(y=microtextures, width=Vanda)
# ax.set_ylabel('Microtextures')
ax.set_xticks(ticks=np.arange(0, 1.25, 0.25))
ax.set_xlabel('Probability of Occurrence')
ax.tick_params(axis='both', which='major', top=True, labeltop=False,
               right=True, labelright=False, labelsize=14)
# Vanda.plot.barh(x=microtextures, y=Vanda)
plt.savefig('Figures/BAREXAMPLE.jpg', bbox_inches='tight', dpi=300)
plt.show()