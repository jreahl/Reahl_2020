#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 10:32:20 2020

@author: jocelynreahl
"""

import numpy as np
import pandas as pd

V_vectors = np.array(([-0.458, 0.788, 0.162, 0.374, 0.045],
                      [0.741, 0.559, 0.231, -0.238, -0.169],
                      [-0.786, -0.003, 0.510, -0.338, 0.087],
                      [0.931, 0.252, -0.049, -0.118, 0.231],
                      [0.645, -0.448, 0.542, 0.297, -0.002]))
p = 5
components = ['PC' + str(x) for x in range(1, p+1)]
variables = ['Mean width', 'Mean depth', 'Current velocity', 'Conductivity',
             'Suspended Matter']
V_vectors = pd.DataFrame(V_vectors, index=variables, columns=components)
V_squared = np.square(V_vectors)
V_squared = V_squared.reset_index()
V_vectors = V_vectors.reset_index()
V_vectors_melt = pd.melt(V_vectors, id_vars='index')
V_squared_melt = pd.melt(V_squared, id_vars='index')
# V_squared = pd.DataFrame(V_squared, columns=components)
b_k = np.zeros(p)
total = np.zeros(p)
for i in range(p):
    for j in range(i+1, p+1):
        total[i] += 1/j
    b_k[i] = (1/p)*total[i]
sorted_df = pd.DataFrame(columns=V_squared_melt.columns)
for var in variables:
    df = V_squared_melt[V_squared_melt['index'] == var].sort_values(
        by=['value', 'variable'], ascending=False)
    sorted_df = sorted_df.append(df)

V_vectors_melt = V_vectors_melt.reindex(sorted_df.index)


    