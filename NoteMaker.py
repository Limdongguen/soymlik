# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 22:37:57 2017

@author: CHOIHEE
"""
import librosa
import random
import pandas as pd
import numpy as np
#import pygame
from sklearn.cluster import KMeans
#from sklearn.decomposition import KMeans


y, sr = librosa.load('AppData/Local/Programs/Python/Python36/RhythmGame/ROOKIE.wav')

"""
설정 부분 - analsis_name(H = 하모니, P =리듬 , Y =원래)
note_num 노트 종류의 갯수
NANIDO = 난이도 조절 (1~10)
group_size 난이도 조절을 위한 그룹 사이즈 (보통 100~1000)
"""

analsis_name = "Y" 
note_num = 4
NANIDO = 10
group_size = 100


if (analsis_name == "H"):
    D_harmonic2, D_percussive2 = librosa.effects.hpss(y, margin=8)
    onset = librosa.onset.onset_detect(y=D_harmonic2, sr=sr)
    D = librosa.stft(D_harmonic2)
elif (analsis_name == "D"):
    D_harmonic2, D_percussive2 = librosa.effects.hpss(y, margin=8)
    onset = librosa.onset.onset_detect(y=D_percussive2, sr=sr)
    D = librosa.stft(D_percussive2)
elif (analsis_name == "Y"):
    onset = librosa.onset.onset_detect(y=y, sr=sr)
    D = librosa.stft(y)

TIME = librosa.frames_to_time(onset, sr=sr)


Ddb = librosa.amplitude_to_db(D, ref=np.max)
AVG = sum(Ddb,0.0)/len(Ddb)


group_num = int(len(D[1])/group_size +1)

onset_group = []
for i in range(group_num):
    tmp = []
    size =  len(onset)
    for j in range(size):
        if (onset[j] < (i*group_size)+group_size):
            tmp.append(onset[j])
        elif(onset[j] >= (i*group_size)+group_size):
            onset = onset[j:size]
            break
    onset_group.append(tmp)

groups = []
tmps = []
for i in range(group_num):
    size = len(onset_group[i])
    tmp = []
    
    for j in range(size):
        tmp.append(AVG[onset_group[i][j]])
    tmps.append(tmp)

groups.append(tmps)
groups.append(onset_group)



notes = []

for i in range(group_num):
    
    nn = int(group_num*NANIDO/100)
    tmp = []
    
    if (nn > (len(groups[0][i]))):
        NANIDO_num =(len(groups[0][i]))
    else :
        NANIDO_num = nn
    for k in range(NANIDO_num):
        max = 1000
        max_index = 0
        size = len(groups[0][i])
        for j in range(size-k):
            if (abs(groups[0][i][j]) < max):
                max = abs(groups[0][i][j])
                max_index = j
        tmp.append(groups[1][i][max_index])
        del groups[1][i][max_index]
        del groups[0][i][max_index]
    notes.append(tmp)

TIME = []
for i in range(group_num):
    
    notes[i].sort()
    TIME.append( librosa.frames_to_time(notes[i], sr=sr))


if (analsis_name == "H"):
    mfcc = librosa.feature.mfcc(y=D_harmonic2, sr=sr)
elif (analsis_name == "D"):
    mfcc = librosa.feature.mfcc(y=D_percussive2, sr=sr)
elif (analsis_name == "Y"):
    mfcc = librosa.feature.mfcc(y=y, sr=sr)

mfcc_T = mfcc
MDF = mfcc_T.T

KK = []
NOTE =[]

for i in range(group_num):
    size = len(TIME[i])
    tmp =[]
    for j in range(size):
        tmp.append(MDF[notes[i][j]])
    KK.append(tmp)

kmeans = []

for i in range(group_num):
    tmp = []
    if(len(KK[i]) >= note_num):
        TMP = KMeans(n_clusters=note_num, random_state=0).fit(KK[i])
        for j in range(len(TMP.labels_)):
            tmp.append(TMP.labels_[j])
    else:
        for j in range(len(KK[i])):
            tmp.append(random.randint(0,note_num-1))
    kmeans.append(tmp)
        
for i in range(group_num):
    size = len(kmeans[i])
    for j in range(size):
        NOTE.append(kmeans[i][j])

tt = []
for i in range(group_num):
    size = len(TIME[i])
    for j in range(size):
        tt.append(TIME[i][j])

out_file = open("C:/Users/CHOIHEE/Desktop/new1.txt", "w")

for n in range(note_num):
    out_file.write("#"+str(n+1)+"\n")
    for i in range(len(tt)):
            if (NOTE[i] ==n):
                out_file.write(str(tt[i])+"\n")


out_file.write("E\n")
out_file.close()      


"""
긴노트 부분
percen = 0.9

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

percen = 0.5
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



"""




