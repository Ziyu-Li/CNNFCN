#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 12:12:39 2020

@author: ziyuli
"""

# model visualization
from keras.models import load_model
model = load_model('/Users/ziyuli/research/AE/germanate/MltTrcDetect.hdf')
from keras.utils import plot_model
plot_model(model,to_file='cnn.png')