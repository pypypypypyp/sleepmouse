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

"""
def csv_reader(name, index, offset=0):
  with open(name, 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    if(isinstance(index, int):
      ret = []
      cnt = 0
      for row in reader:
        if(cnt >= offset):
          ret.append(row[index])
        cnt += 1
      #ret = [row[index] for row in reader]
    elif(isinstance(index, list):
      ret = [ [] for i in range(len(index)) ]
      cnt = 0
      if(cnt >= offset):
        for row in reader:
          if(cnt >= offset):
            for each_index in index:
              ret[each_index].append(row[each_index])
            cnt += 1
  return ret

"""


def csv_reader(name, index, offset=0, use_numpy=False):
  with open(name, 'r') as f:
    reader = csv.reader(f)
    content = np.array([ row for row in reader ], dtype='float')

  if(isinstance(index, int)):
    if(index == -1):
      ret = np.transpose(content)
    else:
      ret = np.transpose(content)[index]
  elif(isinstance(index, list)):
    ret = np.array([np.transpose(content)[each_index] for each_index in index])
  else:
    ret = np.array([])

  if(use_numpy):
    return ret
  else:
    return list(ret)




def csv_reader_full(name, use_numpy):
  ret = np.transpose(csv_reader(name, -1, 0, True))
  if(use_numpy):
    return ret
  else:
    return list(ret)

if __name__ == '__main__':
  main()
