#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 15:47:45 2020

@author: jocelynreahl
"""

import pandas as pd
import numpy as np

master = pd.read_csv('ALLDATA.csv')
transport = ['Aeolian', 'Fluvial', 'Glacial']
# microtextures = ['af', 'as', 'bb', 'cf', 'ff', 'ls', 'saf', 'slf', 'up',  # Polygenetic
#                  'er', 'vc',  # Percussion
#                  'crg', 'cg', 'dt', 'sg',  # High-stress
#                  'de', 'pf',  # Chemicals
#                  'low', 'med', 'high']  # General (applies to all grains)
microtextures = ['af', 'as', 'bb', 'cf', 'crg', 'cg', 'dt', 'de', 'er', 'ff',
                 'ls', 'pf', 'saf', 'sg', 'slf', 'up', 'vc', 'low', 'med', 'high']
transport_original = [master[(master['transport'] == t) & (master['relage'] == 'Active')] for t in transport]
transport_ngrains = [t.loc[:, 'Ngrains'] for t in transport_original]
transport_N = [np.sum(t) for t in transport_ngrains]
transport_data = [t.loc[:, 'af':'high'] for t in transport_original]
transport_data = [t.reindex(columns=microtextures) for t in transport_data]
transport_K = [pd.Series(0, index=microtextures) for t in range(len(transport_data))]
transport_P = [pd.Series(0, index=microtextures) for t in range(len(transport_data))]
for t in range(len(transport_data)):
    for c in range(len(microtextures)):
        for i in range(len(transport_data[t])):
            transport_K[t].iloc[c] += transport_ngrains[t].iloc[i] * transport_data[t].iloc[i, c]
        transport_P[t].iloc[c] = transport_K[t].iloc[c]/transport_N[t]
            