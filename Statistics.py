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
microtextures = ['af', 'as', 'bb', 'cf', 'ff', 'ls', 'saf', 'slf', 'up',  # Polygenetic
                 'er', 'vc',  # Percussion
                 'crg', 'cg', 'dt', 'sg',  # High-stress
                 'de', 'pf',  # Chemicals
                 'low', 'med', 'high']  # General (applies to all grains)
transport_data = [master[master['transport'] == t] for t in transport]