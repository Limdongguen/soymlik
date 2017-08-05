# -*- coding: utf-8 -*-

########################################
# librosa를 이용하여 샘플 데이터를 읽어서 
# 성분(mfcc, tempo, beat_frames, beat_times)을 추출
########################################

import librosa

y, sr = librosa.load('data/sample.mp3')

mfcc = librosa.feature.mfcc(y=y, sr=sr)

print (mfcc.shape)

tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

beat_times = librosa.frames_to_time(beat_frames, sr=sr)


####################
# 음원에서 harmonic - 사람 목소리 악기 연주등 고주파 영역 추출
# percussive - 드럼, 비트 등 짧고 연속적이지 않은 영역 추출
####################

import numpy as np

import librosa

hop_length = 512

y_harmonic, y_percussive = librosa.effects.hpss(y)



#################
# 음원에서 보컬만 추출하는 소
################

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import librosa

import librosa.display

y, sr = librosa.load('audio/Cheese_N_Pot-C_-_16_-_The_Raps_Well_Clean_Album_Version.mp3', duration=120)
S_full, phase = librosa.magphase(librosa.stft(y))
idx = slice(*librosa.time_to_frames([30, 35], sr=sr))
plt.figure(figsize=(12, 4))
librosa.display.specshow(librosa.amplitude_to_db(S_full[:, idx], ref=np.max),
                         y_axis='log', x_axis='time', sr=sr)
plt.colorbar()
plt.tight_layout()

S_filter = librosa.decompose.nn_filter(S_full,
                                       aggregate=np.median,
                                       metric='cosine',
                                       width=int(librosa.time_to_frames(2, sr=sr)))

S_filter = np.minimum(S_full, S_filter)

margin_i, margin_v = 2, 10
power = 2

mask_i = librosa.util.softmask(S_filter,
                               margin_i * (S_full - S_filter),
                               power=power)

mask_v = librosa.util.softmask(S_full - S_filter,
                               margin_v * S_filter,
                               power=power)

S_foreground = mask_v * S_full
S_background = mask_i * S_full

plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
librosa.display.specshow(librosa.amplitude_to_db(S_full[:, idx], ref=np.max),
                         y_axis='log', sr=sr)
plt.title('Full spectrum')
plt.colorbar()

plt.subplot(3, 1, 2)
librosa.display.specshow(librosa.amplitude_to_db(S_background[:, idx], ref=np.max),
                         y_axis='log', sr=sr)
plt.title('Background')
plt.colorbar()
plt.subplot(3, 1, 3)
librosa.display.specshow(librosa.amplitude_to_db(S_foreground[:, idx], ref=np.max),
                         y_axis='log', x_axis='time', sr=sr)
plt.title('Foreground')
plt.colorbar()
plt.tight_layout()
plt.show()

#########

##############################
# 난이도 조절을 위해 margin을 다르게 하여 추출해내는 양을 조절할 수 있다. 
###################################
D_harmonic, D_percussive = librosa.effects.hpss(y)
D_harmonic2, D_percussive2 = librosa.effects.hpss(y, margin=2)
D_harmonic4, D_percussive4 = librosa.effects.hpss(y, margin=4)
D_harmonic8, D_percussive8 = librosa.effects.hpss(y, margin=8)
D_harmonic16, D_percussive16 = librosa.effects.hpss(y, margin=16)
librosa.output.write_wav('data/D1_h.wav', D_harmonic, sr)
librosa.output.write_wav('data/D1_p.wav', D_percussive, sr)
librosa.output.write_wav('data/D2_h.wav', D_harmonic2, sr)
librosa.output.write_wav('data/D2_p.wav', D_percussive2, sr)
librosa.output.write_wav('data/D4_h.wav', D_harmonic4, sr)
librosa.output.write_wav('data/D4_p.wav', D_percussive4, sr)
librosa.output.write_wav('data/D8_h.wav', D_harmonic8, sr)
librosa.output.write_wav('data/D8_p.wav', D_percussive8, sr)
librosa.output.write_wav('data/D16_h.wav', D_harmonic16, sr)
librosa.output.write_wav('data/D16_p.wav', D_percussive16, sr)