# coding: utf-8
import numpy as np

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
  return (1.0 - sigmoid(x)) * sigmoid(x)

def softmax(a):
  exp_a = np.exp(a)
  sum_exp_a = np.sum(exp_a)
  y = exp_a / sum_exp_a
  return y

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # Convert train data from on-hot-vector to index
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
