#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 17:51:05 2022

@author: charlie
"""

from tensorflow.keras.models import load_model 

meg = load_model('/home/charlie/Documents/Uni/Exeter - Data Science/MTHM602_Trends_in_data_science_and_AI/Project/code/fire_1.h5')

meg.summary()