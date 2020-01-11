# coding: utf-8
import sys, os
sys.path.append(os.pardir) # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from collections import OrderedDict
from common.layers import *

class MultiLayerNetExtend:
  """
  
  common.multi_layer_net + Dropout, Batch Normalization
  
  Parameters
  ----------
  input_size: size of input
  hidden_size_list: List of number of neurons (e.g. [100,100,100])
  output_size: size of output
  activation: 'relu' or 'sigmoid'
  weight_init_std: Standard deviation of weight (e.g. 0.01)
  weight_decay_lambda: Parameter of weight decay (L2)
  use_dropout: flag to show Dropout is used or not
  dropout_ration : ratio of Dropout
  use_batchNorm: flag to show BN (Batch Normalization) is used or not
  """
  def __init__(self, input_size, hidden_size_list, output_size,
               activation='relu', weight_init_std='relu', weight_decay_lambda=0,
               use_dropout = False, dropout_ration = 0.5, use_batchnorm=False):
    self.input_size = input_size
    self.output_size = output_size
    self.hidden_size_list = hidden_size_list
    self.hidden_layer_num = len(hidden_size_list)
    self.use_dropout = use_dropout
    self.weight_decay_lambda = weight_decay_lambda
    self.use_batchnorm = use_batchnorm
    self.params = {}

    self.__init_weight(weight_init_std)

    activation_layer = {'sigmoid': Sigmoid, 'relu': Relu}
    self.layers = OrderedDict()
    for idx in range(1, self.hidden_layer_num+1):
      self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
                                                self.params['b' + str(idx)])
      # Layer for BN
      if self.use_batchnorm:
        self.params['gamma' + str(idx)] = np.ones(hidden_size_list[idx-1])
        self.params['beta' + str(idx)] = np.zeros(hidden_size_list[idx-1])
        self.layers['BatchNorm' + str(idx)] = BatchNormalization(self.params['gamma' + str(idx)], self.params['beta' + str(idx)])
            
      self.layers['Activation_function' + str(idx)] = activation_layer[activation]()
        
      # Layer for Dropout
      if self.use_dropout:
          self.layers['Dropout' + str(idx)] = Dropout(dropout_ration)

    idx = self.hidden_layer_num + 1
    self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)], self.params['b' + str(idx)])
  
    self.last_layer = SoftmaxWithLoss()

  """Initialize weights
  
  Parameters
  ----------
  weight_init_std: standard deviation
    'relu' or 'he' -> He's initialization
    'sigmoid' or 'xavier' -> Xavier's initialization
  """
  def __init_weight(self, weight_init_std):
    all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
    for idx in range(1, len(all_size_list)):
      scale = weight_init_std
      if str(weight_init_std).lower() in ('relu', 'he'):
        scale = np.sqrt(2.0 / all_size_list[idx-1])
      elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
        scale = np.sqrt(1.0 / all_size_list[idx-1])
      
      self.params['W'+str(idx)] = scale*np.random.randn(all_size_list[idx-1], all_size_list[idx])
      self.params['b'+str(idx)] = np.zeros(all_size_list[idx])

  def predict(self, x, train_flg=False):
    for key, layer in self.layers.items():
      if "Dropout" in key or "BatchNorm" in key:
        x = layer.forward(x, train_flg)
      else:
        x = layer.forward(x)
    
    return x
  
  def loss(self, x, t, train_flg=False):
    y = self.predict(x, train_flg)
    
    weight_decay = 0
    for idx in range(1, self.hidden_layer_num + 2):
      W = self.params['W'+str(idx)]
      weight_decay += 0.5 * self.weight_decay_lambda*np.sum(W**2)
    
    return self.last_layer.forward(y,t) + weight_decay
  
  def accuracy(self, x, t):
    y = self.predict(x, train_flg=False)
    y = np.argmax(y, axis=1)
    if t.ndim != 1 : t=np.argmax(t, axis=1)
    
    accuracy = np.sum(y==t) / float(x.shape[0])
    return accuracy

  def gradient(self, x, t):
    # forward
    self.loss(x, t, train_flg=True)

    # backward
    dout = 1
    dout = self.last_layer.backward(dout)

    layers = list(self.layers.values())
    layers.reverse()
    for layer in layers:
      dout = layer.backward(dout)

    # compute gradient
    grads = {}
    for idx in range(1, self.hidden_layer_num+2):
      grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW + self.weight_decay_lambda * self.params['W' + str(idx)]
      grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db

      # batch normalization
      if self.use_batchnorm and idx != self.hidden_layer_num+1:
          grads['gamma' + str(idx)] = self.layers['BatchNorm' + str(idx)].dgamma
          grads['beta' + str(idx)] = self.layers['BatchNorm' + str(idx)].dbeta

    return grads