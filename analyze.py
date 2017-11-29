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
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d.axes3d import Axes3D
import statsmodels.formula.api as smf
import statsmodels.api as sm

sfreq = 250
segsec = 6
maxfreq = 12

def detrend(wave):
        time = np.array(range(len(wave)))
        p = stats.linregress(time, wave)[:2]
        return wave-p[1]-p[0]*time

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
        x = pyfftw.interfaces.numpy_fft.fft(np.hanning(sfreq*segsec)*detrend(wave))
        return np.array([abs(c)**2 for c in x[:78]])

def debug():
        for i in range(3):
                plt.subplot(3, 1, i+1)
                dat = detrend(mouse1[0, sfreq*(151+i):sfreq*(152+i)])
                plt.plot(np.arange(151+i, 152+i, 1./sfreq), dat)
                plt.xlabel("time (min)")
                plt.ylabel("intensity")
        plt.tight_layout()
        plt.savefig("temp.png")

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
                        plt.figure(figsize=(10,6))
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
                        plt.savefig("test/mouse%d_%d.png"%(i,j), fontsize=18)
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
        print "Collecting data ..."
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
        D_SCORE_MOUSE1 = D_SCORE_MOUSE2 = D_SCORE_MOUSE3 = D_SCORE_MOUSE4 = [0]
        obs1 = find_observation(1)
        obs2 = find_observation(2)
        obs3 = find_observation(3)
        obs4 = find_observation(4)
        B_OBS_MOUSE1 = obs1[0]
        B_OBS_MOUSE2 = obs2[0]
        B_OBS_MOUSE3 = obs3[0]
        B_OBS_MOUSE4 = obs4[0]
        C_OBS_MOUSE1 = obs1[1]
        C_OBS_MOUSE2 = obs2[1]
        C_OBS_MOUSE3 = obs3[1]
        C_OBS_MOUSE4 = obs4[1]
        D_OBS_MOUSE1 = [(258.7, 295.7)]
        D_OBS_MOUSE2 = [(266.5, 297.3)]
        D_OBS_MOUSE3 = [(273.8, 290.5)]
        D_OBS_MOUSE4 = [(274.1, 288.3)]
        A_SCORE_MOUSE1 = [0, 0, 0, 0]
        A_SCORE_MOUSE2 = [0, 0, 0, 0, 0, 0, 0]
        A_SCORE_MOUSE3 = [0, 0, 0]
        A_SCORE_MOUSE4 = [0]
        A_OBS_MOUSE1 = [(0., 4.), (8., 13.), (22., 38.), (40., 44.)]
        A_OBS_MOUSE2 = [(0., 6.), (11., 18.), (20., 24.), (26., 29.), (32.4, 36.), (42., 52.), (54., 57.)]
        A_OBS_MOUSE3 = [(4., 12.), (18., 23.), (34., 44.)]
        A_OBS_MOUSE4 = [(0.0, 12.)]
        print "--------------DATA-------------"
        print "Chamber 1"
        print "OBSERVATION\t\tSCORE"
        for i in range(len(A_OBS_MOUSE1+B_OBS_MOUSE1+C_OBS_MOUSE1+D_OBS_MOUSE1)):
                print (A_OBS_MOUSE1 + B_OBS_MOUSE1 + C_OBS_MOUSE1 + D_OBS_MOUSE1)[i],"\t\t",(A_SCORE_MOUSE1 + B_SCORE_MOUSE1 + C_SCORE_MOUSE1 + D_SCORE_MOUSE1)[i] 
        print "Chamber 2"
        print "OBSERVATION\t\tSCORE"
        for i in range(len(A_OBS_MOUSE2+B_OBS_MOUSE2+C_OBS_MOUSE2+D_OBS_MOUSE2)):
                print (A_OBS_MOUSE2 + B_OBS_MOUSE2 + C_OBS_MOUSE2 + D_OBS_MOUSE2)[i],"\t\t",(A_SCORE_MOUSE2 + B_SCORE_MOUSE2 + C_SCORE_MOUSE2 + D_SCORE_MOUSE2)[i] 
        print "Chamber 3"
        print "OBSERVATION\t\tSCORE"
        for i in range(len(A_OBS_MOUSE3+B_OBS_MOUSE3+C_OBS_MOUSE3+D_OBS_MOUSE3)):
                print (A_OBS_MOUSE3 + B_OBS_MOUSE3 + C_OBS_MOUSE3 + D_OBS_MOUSE3)[i],"\t\t",(A_SCORE_MOUSE3 + B_SCORE_MOUSE3 + C_SCORE_MOUSE3 + D_SCORE_MOUSE3)[i] 
        print "Chamber 4"
        print "OBSERVATION\t\tSCORE"
        for i in range(len(A_OBS_MOUSE4+B_OBS_MOUSE4+C_OBS_MOUSE4+D_OBS_MOUSE4)):
                print (A_OBS_MOUSE4 + B_OBS_MOUSE4 + C_OBS_MOUSE4 + D_OBS_MOUSE4)[i],"\t\t",(A_SCORE_MOUSE4 + B_SCORE_MOUSE4 + C_SCORE_MOUSE4 + D_SCORE_MOUSE4)[i] 
        OBS_MOUSE1 = A_OBS_MOUSE1 + B_OBS_MOUSE1 + C_OBS_MOUSE1 + D_OBS_MOUSE1
        OBS_MOUSE2 = A_OBS_MOUSE2 + B_OBS_MOUSE2 + C_OBS_MOUSE2 + D_OBS_MOUSE2
        OBS_MOUSE3 = A_OBS_MOUSE3 + B_OBS_MOUSE3 + C_OBS_MOUSE3 + D_OBS_MOUSE3
        OBS_MOUSE4 = A_OBS_MOUSE4 + B_OBS_MOUSE4 + C_OBS_MOUSE4 + D_OBS_MOUSE4
        SCORE_MOUSE1 = A_SCORE_MOUSE1 + B_SCORE_MOUSE1 + C_SCORE_MOUSE1 + D_SCORE_MOUSE1
        SCORE_MOUSE2 = A_SCORE_MOUSE2 + B_SCORE_MOUSE2 + C_SCORE_MOUSE2 + D_SCORE_MOUSE2
        SCORE_MOUSE3 = A_SCORE_MOUSE3 + B_SCORE_MOUSE3 + C_SCORE_MOUSE3 + D_SCORE_MOUSE3
        SCORE_MOUSE4 = A_SCORE_MOUSE4 + B_SCORE_MOUSE4 + C_SCORE_MOUSE4 + D_SCORE_MOUSE4
        score0pow = []
        #MOUSE1
        TIME_MOUSE1 = []
        POW_MOUSE1 = []
        SCORES_MOUSE1 = []
        for i in range(segments[0]):
                x = i/float(10)
                for j in range(len(OBS_MOUSE1)):
                        if OBS_MOUSE1[j][0] <= x and x < OBS_MOUSE1[j][1]:
                                TIME_MOUSE1.append(x)
                                SCORES_MOUSE1.append(SCORE_MOUSE1[j])
                                start = sfreq*segsec*i
                                end = sfreq*segsec*(i+1)
                                ps = powspec(mouse1[0, start:end])
                                if j in range(4): score0pow.append(ps)
                                POW_MOUSE1.append(ps)
                                break
        #MOUSE2
        TIME_MOUSE2 = []
        POW_MOUSE2 = []
        SCORES_MOUSE2 = []
        for i in range(segments[1]):
                x = i/float(10)
                for j in range(len(OBS_MOUSE2)):
                        if OBS_MOUSE2[j][0] <= x and x < OBS_MOUSE2[j][1]:
                                TIME_MOUSE2.append(x)
                                SCORES_MOUSE2.append(SCORE_MOUSE2[j])
                                start = sfreq*segsec*i
                                end = sfreq*segsec*(i+1)
                                ps = powspec(mouse2[0, start:end])
                                if j in range(7): score0pow.append(ps)
                                POW_MOUSE2.append(ps)
                                break
        #MOUSE3
        TIME_MOUSE3 = []
        POW_MOUSE3 = []
        SCORES_MOUSE3 = []
        for i in range(segments[2]):
                x = i/float(10)
                for j in range(len(OBS_MOUSE3)):
                        if OBS_MOUSE3[j][0] <= x and x < OBS_MOUSE3[j][1]:
                                TIME_MOUSE3.append(x)
                                SCORES_MOUSE3.append(SCORE_MOUSE3[j])
                                start = sfreq*segsec*i
                                end = sfreq*segsec*(i+1)
                                ps = powspec(mouse3[0, start:end])
                                if j in range(3): score0pow.append(ps)
                                POW_MOUSE3.append(ps)
                                break
        #MOUSE4
        TIME_MOUSE4 = []
        POW_MOUSE4 = []
        SCORES_MOUSE4 = []
        for i in range(segments[3]):
                x = i/float(10)
                for j in range(len(OBS_MOUSE4)):
                        if OBS_MOUSE4[j][0] <= x and x < OBS_MOUSE4[j][1]:
                                TIME_MOUSE4.append(x)
                                SCORES_MOUSE4.append(SCORE_MOUSE4[j])
                                start = sfreq*segsec*i
                                end = sfreq*segsec*(i+1)
                                ps = powspec(mouse4[0, start:end])
                                if j in range(1): score0pow.append(ps)
                                POW_MOUSE4.append(ps)
                                break
        POW = np.array(POW_MOUSE1 + POW_MOUSE2 + POW_MOUSE3 + POW_MOUSE4)
        SCORES = SCORES_MOUSE1 + SCORES_MOUSE2 + SCORES_MOUSE3 + SCORES_MOUSE4
        cmin = min(SCORES.count(0), SCORES.count(1), SCORES.count(2))
        #prepare for PCA
        score0pow = score0pow[:cmin]
        score1pow = []
        score2pow = []
        A_OBS = A_OBS_MOUSE1 + A_OBS_MOUSE2 + A_OBS_MOUSE3 + A_OBS_MOUSE4
        score1num = 0
        score2num = 0
        for i in range(len(SCORES)):
                 if SCORES[i] == 1 and score1num < cmin:
                        score1num += 1
                        score1pow.append(POW[i])
                 elif SCORES[i] == 2 and score2num < cmin: 
                        score2num += 1
                        score2pow.append(POW[i])
        #POW = [POW[i] for i in range(len(SCORES)) if SCORES[i] != 1]
        #SCORES = [SCORES[i] for i in range(len(SCORES)) if SCORES[i] != 1]
        nPOW = score0pow + score1pow + score2pow
        PCA_DATA = np.array(nPOW)
        pca = PCA(n_components = 3)
        pca.fit(PCA_DATA)
        print pca.explained_variance_ratio_
        transformed = pca.transform(PCA_DATA)
        PC1 = transformed[:, 0]
        PC2 = transformed[:, 1]
        PC3 = transformed[:, 2]
        #removal of outliers
        Y = []
        trans0 = []
        trans1 = []
        trans2 = []
        for i in range(460*3):
               if -3e10 < PC1[i] and PC1[i] < 1e10 \
                       and -2e10 < PC2[i] and PC2[i] < 2e10 \
                       and -2e10 < PC3[i] and PC3[i] < 2e10:
                                trans0.append(PC1[i])
                                trans1.append(PC2[i])
                                trans2.append(PC3[i])
                                if i/460 == 0: Y.append(0)
                                elif i/460 == 1: Y.append(1)
                                elif i/460 == 2: Y.append(2)
        #prepare X and Y
        trans0 = np.array(trans0)
        trans1 = np.array(trans1)
        trans2 = np.array(trans2)
        X = np.column_stack((trans0, trans1, trans2)) #add PC1, PC2, PC3 to X
        X = sm.add_constant(X) #add constant to X
        model = sm.OLS(Y, X)
        results = model.fit()
        print results.summary()
        #PCR parameters
        C, alpha, beta, gamma = results.params
        #output awareness scores
        duration = (145., 152.)
        PCR_scores = []

        #to output time-QAS plot
        """
        for i in range(1, 5):
                powspecs = []
                scores = []
                for j in range(segments[i-1]):
                        start = sfreq*segsec*j
                        end = sfreq*segsec*(j+1)
                        exec("ps = powspec(mouse%d[0, start:end])"%i)
                        powspecs.append(ps)
                for j in range(len(powspecs)/100):
                        transformed = pca.transform(powspecs[j*100:j*100+100])
                        scores.append(C + alpha*transformed[:, 0] + beta*transformed[:, 1] + gamma*transformed[:, 2])
                PCR_scores.append(scores)
        os.system("mkdir test3")
        for i in range(4):
                scores = PCR_scores[i]
                print scores
                for j in range(len(scores)):
                        plt.plot(np.arange(j*10, j*10+10, 0.1), scores[j])
                        plt.xlabel("index")
                        plt.ylabel("quantitative awareness score (QAS)")
                        plt.ylim(-1, 3)
                        plt.savefig("test3/%d_%d.png"%(i, j))
                        plt.clf()
        """
        #to output p, q, r
        """
        p = np.array([str(i) for i in PCR_scores[0*460:0*460+460] if -2 < i and i < 3])
        q = np.array([str(i) for i in PCR_scores[1*460:1*460+460] if -2 < i and i < 3])
        r = np.array([str(i) for i in PCR_scores[2*460:2*460+460] if -2 < i and i < 3])
        open("p.csv", "w").write("\n".join(list(p)))
        open("q.csv", "w").write("\n".join(list(q)))
        open("r.csv", "w").write("\n".join(list(r)))
        print p, q, r
        """
        #to plot nPOW
        """
        os.system("mkdir test2")
        for i in range(len(nPOW)):
                plt.plot(np.arange(0, nPOW[i].size)/6., nPOW[i])
                plt.xlabel("freq (Hz)")
                plt.ylabel("intensity")
                plt.ylim(0, 3.0e10)
                plt.savefig("test2/%d_%d.png"%(i/cmin, i%cmin))
                plt.clf()
        #to output the results of PCA
        """
        """lim = 3.2e10
        for i in range(3):
                plt.plot(np.arange(0, pca.components_[i].size)/6., pca.components_[i])
                plt.xlabel("freq (Hz)")
                plt.ylabel("intensity")
                plt.ylim(-0.6, 0.7)
                plt.savefig("PCA_component%d.png"%(i+1))
                plt.clf() 
        fig = plt.figure()
        ax = Axes3D(fig)
        for i in range(cmin*3):
                if i/cmin == 0:
                        ax.scatter(transformed[i, 0], transformed[i, 1], transformed[i, 2], c="blue", s=3)
                elif i/cmin == 1:
                        ax.scatter(transformed[i, 0], transformed[i, 1], transformed[i, 2], c="yellow", s=3)
                else:
                        ax.scatter(transformed[i, 0], transformed[i, 1], transformed[i, 2], c="red", s=3)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        plt.savefig("PCA.png")
        plt.clf()
        for i in range(cmin*3):
                if i/cmin == 0:
                        plt.scatter(transformed[i, 0], transformed[i, 1], c="blue", s=3)
                elif i/cmin == 1:
                        plt.scatter(transformed[i, 0], transformed[i, 1], c="yellow", s=3)
                else:
                        plt.scatter(transformed[i, 0], transformed[i, 1], c="red", s=3)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.xlim(-lim, lim)
        plt.ylim(-lim, lim)
        plt.savefig("PAC_12.png")
        plt.clf()
        for i in range(cmin*3):
                if i/cmin == 0:
                        plt.scatter(transformed[i, 0], transformed[i, 2], c="blue", s=3)
                elif i/cmin == 1:
                        plt.scatter(transformed[i, 0], transformed[i, 2], c="yellow", s=3)
                else:
                        plt.scatter(transformed[i, 0], transformed[i, 2], c="red", s=3)
        plt.xlabel("PC1")
        plt.ylabel("PC3")
        plt.xlim(-lim, lim)
        plt.ylim(-lim, lim)
        plt.savefig("PAC_13.png")
        plt.clf()
        for i in range(cmin*3):
                if i/cmin == 0:
                        plt.scatter(transformed[i, 1], transformed[i, 2], c="blue", s=3)
                elif i/cmin == 1:
                        plt.scatter(transformed[i, 1], transformed[i, 2], c="yellow", s=3)
                else:
                        plt.scatter(transformed[i, 1], transformed[i, 2], c="red", s=3)
        plt.xlabel("PC2")
        plt.ylabel("PC3")
        plt.xlim(-lim, lim)
        plt.ylim(-lim, lim)
        plt.savefig("PAC_23.png")
        """
if "--output" in sys.argv: output_images()
if "--analyze" in sys.argv: analyze()
if "--debug" in sys.argv: debug()
