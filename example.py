# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 22:37:57 2017

@author: lim dongguen
"""
import librosa
import pandas as pd
import numpy as np
#import pygame

#from sklearn.decomposition import KMeans
from sklearn.cluster import KMeans

y, sr = librosa.load('AppData/Local/Programs/Python/Python36/RhythmGame/1.wav')

D_harmonic2, D_percussive2 = librosa.effects.hpss(y, margin=2)

onset_frames_h2 = librosa.onset.onset_detect(y=D_harmonic2, sr=sr)
onset_frames_p2 = librosa.onset.onset_detect(y=D_percussive2, sr=sr)
TIME_h2 = librosa.frames_to_time(onset_frames_h2, sr=sr)
TIME_p2 = librosa.frames_to_time(onset_frames_p2, sr=sr)



D = librosa.stft(y)
Ddb = librosa.amplitude_to_db(D, ref=np.max)
AVG = sum(Ddb,0.0)/len(Ddb)

DB_h2 = []
DB_p2 = []
DB_h2_INDEX = []
DB_p2_INDEX = []
Onset_h2 = []
Onset_p2 = []
Onset_h2.extend(onset_frames_h2);
Onset_p2.extend(onset_frames_p2);

for i in range(len(onset_frames_h2)):
    DB_h2.append(AVG[onset_frames_h2[i]])
for i in range(len(onset_frames_p2)):
    DB_p2.append(AVG[onset_frames_p2[i]])

NANIDO = 7
NANIDO_H = int(len(Onset_h2)/10*NANIDO)
NANIDO_P = int(len(Onset_p2)/10*NANIDO)

for i in range(NANIDO_H):
    MAX = 1000
    MAX_INDEX = -1
    for j in range(0,len(DB_h2)):
        if abs(DB_h2[j]) <  MAX:
            MAX = float(abs(DB_h2[j]))
            MAX_INDEX = int(j)
    DB_h2_INDEX.append(Onset_h2[MAX_INDEX])
    del DB_h2[MAX_INDEX]
    Onset_h2 = np.delete(Onset_h2,MAX_INDEX)
    
for i in range(NANIDO_P):
    MAX = 1000
    MAX_INDEX = -1
    for j in range(0,len(DB_p2)):
        if abs(DB_p2[j]) <  MAX:
            MAX = float(abs(DB_p2[j]))
            MAX_INDEX = int(j)
    DB_p2_INDEX.append(Onset_p2[MAX_INDEX])
    del DB_p2[MAX_INDEX]
    Onset_p2 = np.delete(Onset_p2,MAX_INDEX)

TIME_h2 = librosa.frames_to_time(DB_h2_INDEX, sr=sr)
TIME_p2 = librosa.frames_to_time(DB_p2_INDEX, sr=sr)


mfcc = librosa.feature.mfcc(y=y, sr=sr)

mfcc_T = mfcc

MDF = mfcc_T.T

KY_h2 = []
for i in range(NANIDO_H):
    KY_h2.append(MDF[DB_h2_INDEX[i]])
    
KY_p2 = []
for i in range(NANIDO_P):
    KY_p2.append(MDF[DB_p2_INDEX[i]])
    
kmeans_h2 = KMeans(n_clusters=3, random_state=0).fit(KY_h2)
kmeans_p2 = KMeans(n_clusters=2, random_state=0).fit(KY_p2)

NOTE = []
for i in range(0,5):
    temp = []
    NOTE.append(temp)

DB_p2_INDEX_TMP = []
DB_h2_INDEX_TMP = []
DB_p2_INDEX_TMP.extend(DB_p2_INDEX)
DB_h2_INDEX_TMP.extend(DB_h2_INDEX)


percen = 0.8

for i in range(NANIDO_H):
    if kmeans_h2.labels_[i] == 0:
        NOTE[0].append(TIME_h2[i])
        AVG_90 = abs(AVG[DB_h2_INDEX[i]]) *percen 
        up = 1
        down = -1
        while(1):
            if (i+up) < len(DB_h2_INDEX_TMP):
                if AVG_90 > abs(AVG[DB_h2_INDEX[i]+up]) :
                   # AVG_90 = abs(AVG[DB_p2_INDEX[i+up]])*percen
                    NOTE[0].append((TIME_h2[i] + 0.02319*up))
                    up += 1
                else :
                    break
            else:
                break
        while(1):
            if (i+down) > 0:
                if AVG_90 > abs(AVG[DB_h2_INDEX_TMP[i]+down]) :
                    NOTE[0].append(TIME_h2[i] + 0.02319*down)
                    down -= 1
                else :
                    break
            else :
                break
            
    if kmeans_h2.labels_[i] == 1:
        NOTE[1].append(TIME_h2[i])
        AVG_90 = AVG[DB_h2_INDEX[i]] *percen 
        up = 1
        down = -1
        while(1):
            if (i+up) < len(DB_h2_INDEX):
                if AVG_90 > abs(AVG[DB_h2_INDEX[i]+up]) :
                    NOTE[1].append((TIME_h2[i] + 0.02319*up))
                    up += 1
                else :
                    break
            else:
                break
        while(1):
            if (i+down) > 0:
                if AVG_90 > abs(AVG[DB_h2_INDEX[i]+down]) :
                    NOTE[1].append(TIME_h2[i] + 0.02319*down)
                    down -= 1
                else :
                    break
            else :
                break
        
    if kmeans_h2.labels_[i] == 2:
        NOTE[2].append(TIME_h2[i])
        AVG_90 = AVG[DB_h2_INDEX[i]] *percen 
        up = 1
        down = -1
        while(1):
            if (i+up) < len(DB_h2_INDEX):
                if AVG_90 > abs(AVG[DB_h2_INDEX[i]+up]) :
                    NOTE[2].append((TIME_h2[i] + 0.02319*up))
                    up += 1
                else :
                    break
            else:
                break
        while(1):
            if (i+down) > 0:
                if AVG_90 > abs(AVG[DB_h2_INDEX[i]+down]) :
                    NOTE[2].append(TIME_h2[i] + 0.02319*down)
                    down -= 1
                else :
                    break
            else :
                break

for i in range(NANIDO_P):
    if kmeans_p2.labels_[i] == 0:
        NOTE[3].append(TIME_p2[i])
        AVG_90 = abs(AVG[DB_p2_INDEX[i]]) *percen 
        up = 1
        down = -1
        while(1):
            if (i+up) < len(DB_p2_INDEX) :
                if AVG_90 > abs(AVG[DB_p2_INDEX[i]+up]) :
                    NOTE[3].append((TIME_p2[i] + 0.02319*up))
                    up += 1
                else :
                    break
            else:
                break
        while(1):
            if (i+down) > 0:
                if AVG_90 > abs(AVG[DB_p2_INDEX[i]+down]) :
                    NOTE[3].append(TIME_p2[i] + 0.02319*down)
                    down -= 1
                else :
                    break
            else :
                break
            
    if kmeans_p2.labels_[i] == 1:
        NOTE[4].append(TIME_p2[i])
        AVG_90 = abs(AVG[DB_p2_INDEX[i]]) *percen 
        up = 1
        down = -1
        while(1):
            if (i+up) < len(DB_p2_INDEX) :
                if AVG_90 > abs(AVG[DB_p2_INDEX[i]+up]) :
                    NOTE[4].append(TIME_p2[i] + 0.02319*up)
                    up += 1
                else :
                    break
            else :
                break
        while(1):
            if (i+down) > 0:
                if AVG_90 > abs(AVG[DB_p2_INDEX[i]+down]) :
                    NOTE[4].append(TIME_p2[i] + 0.02319*down)
                    down -= 1
                else :
                    break
            else :
                break

for i in range(0,5):
    NOTE[i].sort()

out_file = open("AppData/Local/Programs/Python/Python36/RhythmGame/new.txt", "w")
out_file.write("#1\n")
for i in range(len(NOTE[0])):
    out_file.write(str(NOTE[0][i])+"\n")
out_file.write("#2\n")
for i in range(len(NOTE[3])):
    out_file.write(str(NOTE[3][i])+"\n")
out_file.write("#3\n")
for i in range(len(NOTE[1])):
    out_file.write(str(NOTE[1][i])+"\n")
out_file.write("#4\n")
for i in range(len(NOTE[4])):
    out_file.write(str(NOTE[4][i])+"\n")
out_file.write("#5\n")
for i in range(len(NOTE[2])):
    out_file.write(str(NOTE[2][i])+"\n")
out_file.close()
