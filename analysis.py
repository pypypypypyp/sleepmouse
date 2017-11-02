import sys, os, csv
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import csv_manager as cm

def main():
  sampling_freq = 250
  data = cm.csv_reader("data/mouse1.csv", 0)
  segment_sec = 6
  #segment_num = sampling_freq * segment_sec
  segment_num = 1024
  total = len(data)
  print("total: " + str(total))
  time = np.array([cnt / sampling_freq for cnt in range(total)])
  mouse = np.asarray(data, dtype=np.float32)
  begin_min = 0
  fft_log = []

  for segment_begin in range(begin_min*60*sampling_freq, total, segment_num):
    segment_end = segment_begin + segment_num
    resp = mouse[segment_begin: segment_end]
    if(len(resp) < segment_num): continue
    detrend = signal.detrend(resp)
    hanning = detrend * np.hanning(len(detrend)) #in case len(detrend) < segment_num
    fft = np.fft.rfft(hanning)
    absolute = np.abs(fft)
    flow = np.square(absolute / 300000)
    freq = np.fft.rfftfreq(segment_num, 1.0/sampling_freq)

    plt.subplot(2,1,1)
    plt.plot(freq, flow)
    title = str(segment_begin/sampling_freq/60) + 'm' + str(segment_begin/sampling_freq%60) + 's_' +  str(segment_end/sampling_freq/60) + 'm' + str(segment_end/sampling_freq%60) + 's'
    plt.title(title)
    plt.xlabel('freq[Hz]')
    plt.ylabel('Intensity')
    plt.xlim(xmax=40)
    plt.ylim(ymax=1.5)

    plt.subplot(2,1,2)
    plt.plot(np.linspace(segment_begin/sampling_freq, segment_end/sampling_freq, len(resp)), resp)
    plt.title("resp")
    plt.xlabel('time[sec]')
    plt.ylabel('P')
    plt.tight_layout()

    plt.savefig('test/%s.png' % title)
    print('saving figure%d' %segment_begin)
    plt.clf()

    """
    if(cnt==0):
      fft_log.append([segment_begin, absolute, 0])
    else:    
      fft_log.append([segment_begin, absolute, np.corrcoef(fft_log[cnt -1][1],absolute)[0,1]])
    """

    fft_log.append([segment_begin, absolute])


  with open("data/full.csv", 'w') as f:
    #f.write("segment_begin, absolute\n")
    for row in fft_log:
      all_absolute_data = ', '.join(map(str,row[1]))
      f.write("%s, %s\n" % (row[0], all_absolute_data))

if __name__ == '__main__':
  main()
