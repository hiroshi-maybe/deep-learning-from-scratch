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

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
  """Reshape to 2d matrix to forward in convolutional NN

  Parameters
  ----------
  input_data : (batch size, channel, height, width)
  filter_h : height of the filter
  filter_w : width of the filter
  stride : stride size
  pad : padding size

  Returns
  -------
  res : 2d array
  """
  N,C,H,W = input_data.shape
  out_h = (H+2*pad-filter_h)//stride+1
  out_w = (W+2*pad-filter_w)//stride+1

  # pad input_data in vertical and horizontal axis
  img = np.pad(input_data, [(0,0), (0,0), (pad,pad), (pad,pad)], 'constant')
  res = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

  for y in range(filter_h):    
    y_max = y + stride*out_h
    for x in range(filter_w):
        x_max = x + stride*out_w
        res[:,:,y,x,:,:] = img[:,:,y:y_max:stride, x:x_max:stride]    
  res = res.transpose(0,4,5,1,2,3).reshape(N*out_w*out_w, -1)
  return res

def col2im(col, input_syape, filter_h, filter_w, stride=1, pad=0):
  """Reshape to 4d matrix to backward in convolutional NN

  Parameters
  ----------
  col : 2d matrix to reshape
  input_shape: shape of input
  filter_h : height of the filter
  filter_w : width of the filter
  stride: stride size
  pad : paddingsize

  Returns
  -------
  img
  """

  N,C,H,W = input_shape
  out_h = (H+2*pad-filter_h)//stride+1
  out_w = (W+2*pad-filter_w)//stride+1
  col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0,3,4,5,1,2)

  img = np.zeros((N,C,H+2*pad_stride-1, W+2*pad+stride-1))
  for y in range(filter_h):
    y_max = y+stride*out_h
    for x in range(filter_w):
      x_max = x+stride*out_w
      img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]
  
  return img[:, :, pad:H+pad, pad:W+pad]
