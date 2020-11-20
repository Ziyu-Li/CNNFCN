#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 14:18:22 2020

@author: ziyuli
"""

# this code is to use multi trace to autopick the events
from tensorflow.keras import backend
import sys,getopt
opts,args = getopt.getopt(sys.argv[1:],"i:n:o:D:P:d:p:l:t:")
# input a list of event that want to pick
# fo is the output file
# n is the number of the transducer
# D is the detect model
# P is the pick model in a style
for o,a in opts:
    if o in ("-i"):
        lstn = a
    if o in ("-o"):
        fo = a
    if o in ("-n"):
        n_st = int(a)
    if o in ("-D"):
        detect = a
    if o in ("-P"):
        pick = a
    if o in ("-d"):
        d_threshold = float(a)
    if o in ("-p"):
        p_threshold = float(a)
    if o in ("-l"):
        unit = int(a)
    if o in ("-t"):
        t_dif = float(a)

# load model
from tensorflow.keras.models import load_model
pickmodel = load_model(pick)
detectmodel = load_model(detect)

# load data
import numpy as np
fi = open(lstn,'r')
eve = []
for line in fi:    
    eve.append(line.split('\n')[0])
    
from obspy import read
n_eve = int(len(eve)/n_st)
x_total = None
for i in range(0,n_eve):
    x_eve = None
    for j in range(0,n_st):
        info = read(eve[i*n_st+j])
        dt = info[0].stats.delta
        amp=np.array(info[0].data)
        amp = amp.reshape(1,amp.shape[0],1)
        if x_eve is None:
            x_eve = amp
        else:
            x_eve = np.append(x_eve,amp,axis=2)
    if x_total is None:
        x_total = x_eve
    else:
        x_total = np.append(x_total,x_eve,axis=0)

# begin to put the data into the model

l =8
t = []
n_dif = int(t_dif/dt)
for i in range(0,x_total.shape[0]):
    k=0
    flg = 0
    while k < int(l):
    
        x_input = x_total[i,k*unit-2*flg*n_dif:(k+1)*unit-2*flg*n_dif,:].reshape(1,unit,n_st)
        for s in range(0,n_st):
            x = np.mean(x_input[0,:,s])
            x_junk = x_input[0,:,s]-x
            x_junk1 = x_junk/max(np.abs(x_junk))
            x_input[0,:,s] = x_junk1
        x_input= x_input.reshape(x_input.shape[0],x_input.shape[1],x_input.shape[2],1)
        y = detectmodel.predict(x_input)
        if y > d_threshold:
            ypick = pickmodel.predict(x_input)

    
                                   
            t0 = []
            psb = []
            maxn = []

            for s in range(0,n_st):
                maxn.append(np.argmax(ypick[0,:,s,0]))
                t0.append((np.argmax(ypick[0,:,s,0])+k*unit-2*flg*n_dif)*dt)
                psb.append((max(ypick[0,:,s,0])))
            if min(psb) > p_threshold:
            
                if min(maxn)>unit-n_dif:
                    flg = 1
                
                else:  
                    flg = 0 
                    for s in range(0,n_st):
                        print(eve[i*n_st+s],t0[s],y[0,0],psb[s])
                    
        k=k+1
             
                
                
        
                
