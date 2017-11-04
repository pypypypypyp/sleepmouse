#!/usr/bin/python
#coding: utf-8

import os
import sys
import cv2
from math import *
import numpy as np
from scipy import fftpack, stats, signal
import pyfftw
from matplotlib import pyplot as plt
from progressbar import ProgressBar, Percentage, Bar
import time

sfreq = 250
segsec = 6
maxfreq = 12

def detrend(seg):
        time = np.array(range(len(seg)))
        p = stats.linregress(time, seg)[:2]
        return seg-p[1]-p[0]*time

def lp(data, thres, n_d):
        fftdata = pyfftw.interfaces.numpy_fft.fft(data)
        shiftfftdata = np.fft.fftshift(fftdata)
        mask = np.zeros(shiftfftdata.size)
        mask[mask.size/2-int(thres*n_d):mask.size/2+int(thres*n_d)] = 1
        ishiftfftdata = np.fft.ifftshift(shiftfftdata*mask)
        ret = pyfftw.interfaces.numpy_fft.ifft(ishiftfftdata)
        return np.abs(ret)

def time_to_min(timestr):
        spl = timestr.split(":")
        return int(spl[0])*60+int(spl[1])+int(spl[2])/6*6*10/60/float(10)

print "Reading files ..."

start = "20171031"
l = [open(s).readlines() for s in os.listdir(os.getcwd()) if s.startswith(start)]
combined = l[0][13:]+l[1][13:]+l[2][13:]+l[3][13:]

combnp = np.asarray([np.asarray([float(num) for num in combined[i].split(",")]) for i in range(len(combined))])
mouse1 = np.array([combnp[:, 0], combnp[:, 1]])
mouse2 = np.array([combnp[:, 2], combnp[:, 3]])
mouse3 = np.array([combnp[:, 4], combnp[:, 5]])
mouse4 = np.array([combnp[:, 6], combnp[:, 7]])

segments = range(4)
segments[0] = len(mouse1[0])/(sfreq*segsec)
segments[1] = len(mouse2[0])/(sfreq*segsec)
segments[2] = len(mouse3[0])/(sfreq*segsec)
segments[3] = len(mouse4[0])/(sfreq*segsec)

os.system("mkdir test")

print "Processing ..."

def read_results():
        #read DRUG_INJECTION_TIME and AWARENESS_SCORE
        b = [line.split("\t") for line in open("B.csv").readlines()]
        c = [line.split("\t") for line in open("C.csv").readlines()]
        b_inject = [] #time when drug is injected
        b_score = [] #awareness score
        for bind in range(len(b)):
                if len(b[bind][0])==1:
                        b_inject.append([time_to_min(j[2]) for j in b[bind:bind+8]])
                        b_score.append([float(j[3]) for j in b[bind:bind+8]])
        c_inject = []
        c_score = []
        for cind in range(len(c)):
                if len(c[cind][0])==1:
                        c_inject.append([time_to_min(j[2]) for j in c[cind:cind+8]])
                        c_score.append([float(j[3]) for j in c[cind:cind+8]])
        return b_inject, b_score, c_inject, c_score

def boxopen(mouse_id):
        openbox = []
        for i in range(segments[mouse_id-1]):
                start = sfreq*segsec*i
                end = sfreq*segsec*(i+1)
                exec("tempD = detrend(mouse%d[0, start:end])"%mouse_id)
                if np.where(np.abs(tempD)>4000)[0].size>50:
                        if i*0.1>60: openbox.append(i*0.1)
        return openbox

def powspec(wave):
        x = pyfftw.interfaces.nnumpy_fft.fft(np.hanning(sfreq*segsec)*detrend(wave))
        return np.array([abs(c)**2 for c in x[:sfreq/10*segsec/2+1]])

def output_images():
        for i in range(1, 5):
                p = ProgressBar(widgets=[Percentage(), Bar()], maxval=100).start()
                peaks = range(segments[i-1])
                for j in range(segments[i-1]):
                        start = sfreq*segsec*j
                        end = sfreq*segsec*(j+1)
                        exec("tempD = detrend(mouse%d[0, start:end])"%i)
                        tempH = tempD*np.hanning(sfreq*segsec)
                        x = pyfftw.interfaces.numpy_fft.fft(tempH)
                        freq = np.fft.fftfreq(sfreq*segsec, d=1./sfreq)
                        amp = np.array([abs(c)**2 for c in x[:sfreq/10*segsec/2+2]])
                        freq = freq[:amp.size]
                        temp = sorted([[k, amp[k]] for k in signal.argrelmax(amp)[0] if amp[k]>np.max(amp)*0.2], key=lambda x:x[1])
                        if len(temp)>=4: peaks[j]=[z[0] for z in temp[-4:]]
                        else: peaks[j]=[z[0] for z in temp]
                        plt.figure(figsize=(30,20))
                        #respiration
                        h1=plt.subplot(3, 1, 1)
                        plt.plot(np.arange(0, 6, 1./sfreq), tempD)
                        plt.xlabel("time(sec)")
                        plt.ylabel("resp")
                        plt.ylim(-4000, 4000)
                        #power spectrum
                        h2=plt.subplot(3, 1, 2)
                        plt.plot(freq, amp)
                        plt.xlabel("freq(Hz)")
                        plt.ylabel("I")
                        #peak
                        h3=plt.subplot(3, 1, 3)
                        [plt.scatter([z%100]*len(peaks[z]), np.array(peaks[z])/float(segsec), c="red") for z in range(j/100*100, j+1)]
                        plt.xlim(0, 100)
                        plt.ylim(0, 12)
                        plt.xlabel("time(*0.1 min)")
                        plt.ylabel("I")
                        plt.tight_layout()
                        plt.savefig("test/mouse%d_%d.png"%(i,j), dpi=50)
                        plt.close()
                        h1.cla()
                        h2.cla()
                        h3.cla()
                        ind = (j+1)*100/segments[i-1]
                        p.update(ind)

def find_observation(mouse_id):
        #After injection, if box is opened at time A, observation starts at A.
        #And 10 minutes after that, if box is opened at time B, observation ends at B.
        bopen = boxopen(mouse_id)
        copen = boxopen(mouse_id)
        b_inject, b_score, c_inject, c_score = read_results()
        b_inject[mouse_id-1].append(b_inject[mouse_id-1][-1]+10)
        c_inject[mouse_id-1].append(c_inject[mouse_id-1][-1]+10)
        if mouse_id==3: b_inject[2][-1]+=5
        if mouse_id==4: b_inject[3][-1]-=5; c_inject[3][-1]+=3
        
        b_obs = []
        for bind in range(len(b_inject[mouse_id-1])-1):
                start = max([i for i in bopen if int(b_inject[mouse_id-1][bind]*10)-30 < int(i*10) and int(i*10) < int(b_inject[mouse_id-1][bind]*10)+30])
                end = min([i for i in bopen if int(b_inject[mouse_id-1][bind+1]*10)-30 < int(i*10) and int(i*10) < int(b_inject[mouse_id-1][bind+1]*10)+30])
                b_obs.append((round(start+0.1), round(end-0.1)))
        c_obs = []
        for cind in range(len(c_inject[mouse_id-1])-1):
                start = max([i for i in copen if int(c_inject[mouse_id-1][cind]*10)-35 < int(i*10) and int(i*10) < int(c_inject[mouse_id-1][cind]*10)+35])
                end = min([i for i in copen if int(c_inject[mouse_id-1][cind+1]*10)-30 < int(i*10) and int(i*10) < int(c_inject[mouse_id-1][cind+1]*10)+30])
                c_obs.append((round(start+0.1), round(end-0.1)))
        return b_obs, c_obs

def analyze():
        b_inject, b_score, c_inject, c_score = read_results()
        A_SCORE_MOUSE1 = A_SCORE_MOUSE2 = A_SCORE_MOUSE3 = A_SCORE_MOUSE4 = [0]
        B_SCORE_MOUSE1 = b_score[0]
        B_SCORE_MOUSE2 = b_score[1]
        B_SCORE_MOUSE3 = b_score[2]
        B_SCORE_MOUSE4 = b_score[3]
        C_SCORE_MOUSE1 = c_score[0]
        C_SCORE_MOUSE2 = c_score[1]
        C_SCORE_MOUSE3 = c_score[2]
        C_SCORE_MOUSE4 = c_score[3]
        obs1 = find_observation(1)
        obs2 = find_observation(2)
        obs3 = find_observation(3)
        obs4 = find_observation(4)
        A_OBS_MOUSE1 = A_OBS_MOUSE2 = A_OBS_MOUSE3 = A_OBS_MOUSE4 = [(0.0, 65.0)]
        B_OBS_MOUSE1 = obs1[0]
        B_OBS_MOUSE2 = obs2[0]
        B_OBS_MOUSE3 = obs3[0]
        B_OBS_MOUSE4 = obs4[0]
        C_OBS_MOUSE1 = obs1[1]
        C_OBS_MOUSE2 = obs2[1]
        C_OBS_MOUSE3 = obs3[1]
        C_OBS_MOUSE4 = obs4[1]
        print "--------------DATA-------------"
        print "Chamber 1"
        print "OBSERVATION\t\tSCORE"
        for i in range(len(A_OBS_MOUSE1+B_OBS_MOUSE1+C_OBS_MOUSE1)):
                print (A_OBS_MOUSE1 + B_OBS_MOUSE1 + C_OBS_MOUSE1)[i],"\t\t",(A_SCORE_MOUSE1 + B_SCORE_MOUSE1 + C_SCORE_MOUSE1)[i] 
        print "Chamber 2"
        print "OBSERVATION\t\tSCORE"
        for i in range(len(A_OBS_MOUSE2+B_OBS_MOUSE2+C_OBS_MOUSE2)):
                print (A_OBS_MOUSE2 + B_OBS_MOUSE2 + C_OBS_MOUSE2)[i],"\t\t",(A_SCORE_MOUSE2 + B_SCORE_MOUSE2 + C_SCORE_MOUSE2)[i] 
        print "Chamber 3"
        print "OBSERVATION\t\tSCORE"
        for i in range(len(A_OBS_MOUSE3+B_OBS_MOUSE3+C_OBS_MOUSE3)):
                print (A_OBS_MOUSE3 + B_OBS_MOUSE3 + C_OBS_MOUSE3)[i],"\t\t",(A_SCORE_MOUSE3 + B_SCORE_MOUSE3 + C_SCORE_MOUSE3)[i] 
        print "Chamber 4"
        print "OBSERVATION\t\tSCORE"
        for i in range(len(A_OBS_MOUSE4+B_OBS_MOUSE4+C_OBS_MOUSE4)):
                print (A_OBS_MOUSE4 + B_OBS_MOUSE4 + C_OBS_MOUSE4)[i],"\t\t",(A_SCORE_MOUSE4 + B_SCORE_MOUSE4 + C_SCORE_MOUSE4)[i] 

if "--output" in sys.argv: output_images()
if "--analyze" in sys.argv: analyze()

