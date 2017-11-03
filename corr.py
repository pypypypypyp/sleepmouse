import sys, os
import numpy as np
import csv_manager as cm
import matplotlib.pyplot as plt

def main():
  csv_data1 = cm.csv_reader("proccessed/mouse1/full.csv", -1, 0, True)
  csv_data2 = cm.csv_reader("proccessed/mouse2/full.csv", -1, 0, True)
  label = csv_data1[0]
  abs_data1 = csv_data1[1:]
  abs_data2 = csv_data2[1:]
  l = len(abs_data1)

  corr_val = np.corrcoef(abs_data1, abs_data2)
  draw_heatmap(corr_val[:l,:l], label[:l], label[:l])


def draw_heatmap(data, row_labels, column_labels):
  #copy from  https://qiita.com/ynakayama/items/7dc01f45caf6d87a981b
  fig, ax = plt.subplots()
  heatmap = ax.pcolor(data, cmap=plt.cm.Blues)

  ax.set_xticks(np.arange(data.shape[0]) + 0.5, minor=False)
  ax.set_yticks(np.arange(data.shape[1]) + 0.5, minor=False)

  ax.invert_yaxis()
  ax.xaxis.tick_top()

  ax.set_xticklabels(row_labels, minor=False)
  ax.set_yticklabels(column_labels, minor=False)
  plt.show()
  #plt.savefig('image.png')
  return heatmap


if (__name__ == "__main__"):
  main()
