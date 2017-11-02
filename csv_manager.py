import sys, os, csv
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def main():
  sampling_freq = 250
  ret = get_data_from_csv()
  cnt = 0
  for mouse in ret:
    cnt += 1
    with open('data/mouse' + str(cnt) + '.csv', 'w') as f:
      for row in mouse:
        f.write(str(row) + '\n')


def get_data_from_csv():
  filelist = ['1','2','3','4']
  ret = []
  for i in range(4):
    mouse = []
    for filename in filelist:
      mouse.extend(csv_reader('data/' + filename + '.csv', i*2, 13))
    ret.append(np.array(mouse))

  # return [mouse1, mouse2, mouse3]
  return ret

def csv_reader(name, index, offset=0):
  with open(name, 'r') as f:
    reader = csv.reader(f)
    header = next(reader)

    cnt = 0
    ret = []
    for row in reader:
      if(cnt >= offset):
        ret.append(row[index])
      cnt += 1
    #ret = [row[index] for row in reader]
    return ret



if __name__ == '__main__':
  main()
