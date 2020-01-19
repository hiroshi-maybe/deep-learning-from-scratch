import numpy as np

# http://glowingpython.blogspot.jp/2012/02/convolution-with-numpy.html
def smooth_curve(x):
  window_len = 11
  s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
  w = np.kaiser(window_len, 2)
  y = np.convolve(w/w.sum(), s, mode='valid')
  return y[5:len(y)-5]

def shuffle_dataset(x, t):
  """shuffle data set
  Parameters
  ----------
  x: training data
  t: test data
  
  Returns
  -------
  x, t: shuffled training data and test data
  """
  permutation = np.random.permutation(x.shape[0])
  x = x[permutation,:] if x.ndim==2 else x[permutation,:,:,:]
  t = t[permutation]

  return x,t
