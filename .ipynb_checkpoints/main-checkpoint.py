# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np


def forward_sub(L,b):
    L = np.copy(L)
    b = np.copy(b)
    for i in range(len(L)):
        b[i] = b[i] / L[i,i]
        L[i][i] = 1
        for j in range(i+1,len(L)):
            b[j] = b[j] - b[i] * (L[j][i])
            L[j] = L[j] - L[i] * (L[j][i])
    return b


def backward_sub(LT,y):
    LT = np.copy(LT)
    y = np.copy(y)
    for i in reversed(range(len(LT))):
        y[i] = y[i] / LT[i][i]
        LT[i][i] = 1
        for j in range(0,i):
            y[j] = y[j] - y[i] * (LT[j][i])
            LT[j] = LT[j] - LT[i] * (LT[j][i])
    return y


def sigma_solver(s,UTb):
    s = np.copy(s)
    UTb = np.copy(UTb)
    y = np.array([UTb[i]/s[i][i] for i in range(len(UTb)-1)])
    return y

if __name__ == '__main__':

  for i in range(3):
      print(i)
  a = np.array([1,2,3,4,5])
  c = a[:3]
  print(c)
  b = a
  b[2] = 5
  print(a)
  print(b)
  print('**********')
  a = np.array([1,2,3])
  b = np.array(a)
  b[2] = 5
  print(a)
  print(b)
#  See PyCharm help at https://www.jetbrains.com/help/pycharm/
