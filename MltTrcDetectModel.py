#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 18:06:44 2020

@author: ziyuli
"""

# here I want to develop a new model for detection
# I used 6 trasducers data together to train my model
'''

n_st = 6

pwd = "/Users/ziyuli/research/AE/germanate/"
# the input will be a list of events with their arrival times
# events should be sorted with the event list
fn = pwd+"detect.lst"
fi = open(fn,'r')
eve = []
arr = []
'''
import sys,getopt
opts,args = getopt.getopt(sys.argv[1:],"i:n:o:l:")
# input a list of event that want to pick
# fo is the output file
# s is the number of the transducer
# D is the detect model
# P is the pick model in a style
for o,a in opts:
    if o in ("-i"):
        fn = a
    if o in ("-o"):
        fo = a
    if o in ("-n"):
        n_st = int(a)
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

x_total = None

for j in range(0,n_eve):
    x_eve = None
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
    
    if x_total is None:
        x_total = x_eve
    else:
        x_total = np.append(x_total,x_eve,axis=0)
        
# prepare for training set 
n_sam = 100

x_train = None
y_train = None
for i in range(0,n_eve):
    mini = int(min(arr[i,0,:])/dt)
    maxi = int(max(arr[i,0,:])/dt)
    amp = x_total[i,:,:]
    if maxi <unit:
        n_ran = random.sample(range(0,mini),int(n_sam/2))
        n_ran.extend(random.sample(range(mini,x_total.shape[1]-unit),n_sam-int(n_sam/2)))
    else:
        n_ran = random.sample(range(maxi-unit,mini),int(n_sam/2))
        n_ran.extend(random.sample(range(0,maxi-unit),int(n_sam/4)))
        n_ran.extend(random.sample(range(maxi,x_total.shape[1]-unit),n_sam-int(n_sam/2)-int(n_sam/4)))
    for l in range(0,n_sam):
        n_i = n_ran[l]
        if x_train is None:
            x_train = x_total[i,n_i:n_i+unit,:]
            x_train = x_train.reshape(1,unit,n_st)
#            x_train = np.abs(x_train)
            for k in range(0,n_st):
                x = max(x_train[0,:,k])
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
    if y_train  is None:
        y_train = np.append(np.ones(int(n_sam/2)),np.zeros(n_sam-int(n_sam/2)))
    else:
        y_tmp = np.append(np.ones(int(n_sam/2)),np.zeros(n_sam-int(n_sam/2)))
        y_train = np.append(y_train,y_tmp)
y_train = y_train.reshape(y_train.shape[0],1)#(5000,1)
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)#(5000,512,6,1)
# model SET UP
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D,MaxPooling2D,BatchNormalization
from keras.layers import UpSampling1D,AveragePooling1D,AveragePooling2D 
from keras.layers import Cropping1D,Cropping2D,ZeroPadding1D,ZeroPadding2D 
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
import keras

batch_size = 128
num_classes = 20 # from 1 to 19, python 0-19 is 20
epochs = 100

detector = Sequential()
detector.add(Conv2D(32,(16,2),input_shape = (unit,n_st,1),activation = 'relu',padding='same'))
detector.add(MaxPooling2D(pool_size=(4,1)))
detector.add(Conv2D(32,(16,2),activation = 'relu',padding='same'))
detector.add(MaxPooling2D(pool_size=(4,1)))
detector.add(Conv2D(32,(16,2),activation = 'relu',padding='same'))
detector.add(MaxPooling2D(pool_size=(4,1)))
detector.add(Flatten())
detector.add(Dense(output_dim=128,activation='relu'))
detector.add(Dense(output_dim=1,activation='sigmoid'))


detector.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


detector.fit([x_train], [y_train],batch_size=batch_size, epochs=epochs)

detector.save(fo)

#detector.save(pwd+'multitraceDetect.hdf')

        
    
    




