#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 19:57:21 2020

@author: ziyuli
"""

# this code is used to build the pick model B
import sys,getopt
opts,args = getopt.getopt(sys.argv[1:],"i:n:o:s:l:")

for o,a in opts:
    if o in ("-i"):
        fn = a
    if o in ("-o"):
        fo = a
    if o in ("-n"):
        n_st = int(a)
    if o in ("-s"):
        sigma = float(a)
    if o in ("-l"):
        unit = int(a)

fi = open(fn,'r')
eve = []
arr = []

import numpy as np
for line in fi:
    a = line.split()[0]
    b = line.split()[1]
    eve.append(a)
    arr.append(float(b))

n_eve = int(len(eve)/n_st)
arr = np.array(arr) 
arr = arr.reshape(n_eve,n_st)
arr = arr.reshape(n_eve,1,n_st)

# Preprocessing data
from obspy import read
import random
import math
from scipy.stats import norm
x_total = None
y_total = None

for j in range(0,n_eve):
    x_eve = None
    y_eve = None
    for m in range(0,n_st):
        i = j*n_st+m
        st = read(eve[i])
        dt = st[0].stats.delta
        amp = np.array(st[0].data)
        amp = amp.reshape(1,amp.shape[0],1)
        if x_eve is None:
            x_eve = amp
        else:
            x_eve = np.append(x_eve,amp,axis=2)
            
        x0 = np.linspace(0,amp.shape[1]-1,amp.shape[1])
        x0 = x0.reshape(1,x0.shape[0])
        n0 = int(arr[j,:,m]/dt)
        y = norm.pdf(x0,n0,sigma/dt)
        y = y/np.max(y)
        y = y.reshape(1,y.shape[1],1)
        if y_eve is None:
            y_eve =y
        else:
            y_eve = np.append(y_eve,y,axis=2)
    
    if x_total is None:
        x_total = x_eve
        y_total = y_eve
    else:
        x_total = np.append(x_total,x_eve,axis=0)
        y_total = np.append(y_total,y_eve,axis=0)
        
# prepare for training set 
n_sam = 100
x_train = None
y_train = None
for i in range(0,n_eve):
    mini = int(min(arr[i,0,:])/dt)
    maxi = int(max(arr[i,0,:])/dt)
    amp = x_total[i,:,:]
    if maxi <unit:
        n_ran = random.sample(range(0,maxi),n_sam)
    else:
        n_ran = random.sample(range(maxi-unit,maxi),n_sam)
    for l in range(0,n_sam):
        n_i = n_ran[l]
        if x_train is None:
            x_train = x_total[i,n_i:n_i+unit,:]
            x_train = x_train.reshape(1,unit,n_st)
#            x_train = np.abs(x_train)
            for k in range(0,n_st):
                x0 = np.mean(x_train[0,:,k])
                x_junk1 = x_train[0,:,k]-x0
                x_junk2 = x_junk1/max(np.abs(x_junk1))
                x_junk = x_train[0,:,k]/max(x_train[0,:,k])
                x_train[0,:,k] = x_junk2
        else:
            x_tmp = x_total[i,n_i:n_i+unit,:]
            x_tmp = x_tmp.reshape(1,unit,n_st)
#            x_tmp = np.abs(x_tmp)
            for k in range(0,n_st):
                x0 = np.mean(x_tmp[0,:,k])
                x_junk1 = x_tmp[0,:,k]-x0
                x_junk2 = x_junk1/max(np.abs(x_junk1))
                x_junk = x_tmp[0,:,k]/max(x_tmp[0,:,k])
                x_tmp[0,:,k] = x_junk2
            x_train = np.append(x_train,x_tmp,axis=0)
        if y_train is None:
            y_train = y_total[i,n_i:n_i+unit,:]
            y_train = y_train.reshape(1,unit,n_st)
        else:
            y_tmp = y_total[i,n_i:n_i+unit,:]
            y_tmp = y_tmp.reshape(1,unit,n_st)
            y_train = np.append(y_train,y_tmp,axis=0)
        
       
y_train = y_train.reshape(y_train.shape[0],y_train.shape[1],y_train.shape[2],1)
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
        
# model SET UP
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D,MaxPooling2D,BatchNormalization
from keras.layers import UpSampling2D,AveragePooling1D,AveragePooling2D 
from keras.layers import Cropping1D,Cropping2D,ZeroPadding1D,ZeroPadding2D 
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
from keras.layers.advanced_activations import LeakyReLU
import keras
batch_size = 128
num_classes = 20 # from 1 to 19, python 0-19 is 20
epochs = 100

pick = Sequential()
# begin downsampling
pick.add(Conv2D(4,(3,3),input_shape = (unit,n_st,1),activation = 'relu',padding='same'))
pick.add(AveragePooling2D(pool_size = (2,1),padding = 'same')) #1
pick.add(LeakyReLU(alpha = 0.2))
pick.add(Dropout(0.25))
pick.add(BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,center=True,scale=True,
beta_initializer='zeros', gamma_initializer='ones',  moving_mean_initializer='zeros',
moving_variance_initializer='ones'))
pick.add(Conv2D(8,(3,3),padding='same'))
pick.add(AveragePooling2D(pool_size = (2,1),padding = 'same')) #2
pick.add(LeakyReLU(alpha = 0.2))
pick.add(Dropout(0.25))
pick.add(BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,center=True,scale=True,
beta_initializer='zeros', gamma_initializer='ones',  moving_mean_initializer='zeros',
moving_variance_initializer='ones'))
pick.add(Conv2D(16,(3,3),padding='same'))
pick.add(AveragePooling2D(pool_size = (2,1),padding = 'same')) #3
pick.add(LeakyReLU(alpha = 0.2))
pick.add(Dropout(0.25))
pick.add(BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,center=True,scale=True,
beta_initializer='zeros', gamma_initializer='ones',  moving_mean_initializer='zeros',
moving_variance_initializer='ones'))
pick.add(Conv2D(32,(3,3),padding='same'))
pick.add(AveragePooling2D(pool_size = (2,1),padding = 'same')) #4
pick.add(LeakyReLU(alpha = 0.2))
pick.add(Dropout(0.25))
pick.add(BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,center=True,scale=True,
beta_initializer='zeros', gamma_initializer='ones',  moving_mean_initializer='zeros',
moving_variance_initializer='ones'))
pick.add(Conv2D(64,(3,3),padding='same'))
pick.add(AveragePooling2D(pool_size = (2,1),padding = 'same')) #5
pick.add(LeakyReLU(alpha = 0.2))
pick.add(Dropout(0.25))
pick.add(BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,center=True,scale=True,
beta_initializer='zeros', gamma_initializer='ones',  moving_mean_initializer='zeros',
moving_variance_initializer='ones'))

# began upsampling
pick.add(Conv2D(128,(3,3),padding='same'))
pick.add(UpSampling2D(size = (2,1))) #1
pick.add(LeakyReLU(alpha = 0.2))
pick.add(Dropout(0.25))
pick.add(BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,center=True,scale=True,
beta_initializer='zeros', gamma_initializer='ones',  moving_mean_initializer='zeros',
moving_variance_initializer='ones'))
pick.add(Conv2D(64,(3,3),padding='same'))
pick.add(UpSampling2D(size = (2,1))) #2
pick.add(LeakyReLU(alpha = 0.2))
pick.add(Dropout(0.25))
pick.add(BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,center=True,scale=True,
beta_initializer='zeros', gamma_initializer='ones',  moving_mean_initializer='zeros',
moving_variance_initializer='ones'))
pick.add(Conv2D(32,(3,3),padding='same'))
pick.add(UpSampling2D(size = (2,1))) #3
pick.add(LeakyReLU(alpha = 0.2))
pick.add(Dropout(0.25))
pick.add(BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,center=True,scale=True,
beta_initializer='zeros', gamma_initializer='ones',  moving_mean_initializer='zeros',
moving_variance_initializer='ones'))
pick.add(Conv2D(16,(3,3),padding='same'))
pick.add(UpSampling2D(size = (2,1))) #4
pick.add(LeakyReLU(alpha = 0.2))
pick.add(Dropout(0.25))
pick.add(BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,center=True,scale=True,
beta_initializer='zeros', gamma_initializer='ones',  moving_mean_initializer='zeros',
moving_variance_initializer='ones'))
pick.add(Conv2D(8,(3,3),padding='same'))
pick.add(UpSampling2D(size = (2,1))) #5
pick.add(LeakyReLU(alpha = 0.2))
pick.add(Dropout(0.25))
pick.add(BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,center=True,scale=True,
beta_initializer='zeros', gamma_initializer='ones',  moving_mean_initializer='zeros',
moving_variance_initializer='ones'))
pick.add(Conv2D(4,(3,3),padding='same'))
pick.add(LeakyReLU(alpha = 0.2))
pick.add(Dropout(0.25))
pick.add(BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,center=True,scale=True,
beta_initializer='zeros', gamma_initializer='ones',  moving_mean_initializer='zeros',
moving_variance_initializer='ones'))
pick.add(Conv2D(1,(3,3),padding='same'))

optimizer = keras.optimizers.Adadelta(lr=1,rho=0.95,epsilon=1e-06)
pick.compile(optimizer=optimizer,loss='mse',metrics=['accuracy'])
pick.fit([x_train], [y_train],batch_size=batch_size, epochs=epochs)

pick.save(fo)












