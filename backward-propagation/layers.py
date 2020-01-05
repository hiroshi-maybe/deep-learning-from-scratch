import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import *

class MultiLayer:
  def __init__(self):
    self.x = None
    self.y = None
  
  def forward(self, x, y):
    self.x = x
    self.y = y
    out = x*y
    
    return out

  def backward(self, dout):
    dx = dout * self.y
    dy = dout * self.x
    
    return dx,dy

class AddLayer:
  def __init__(self):
    pass
  
  def forward(self, x, y):
    out = x+y
    return out
    
  def backward(self, dout):
    dx = dy = dout * 1
    return dx, dy

class Relu:
  def __init__(self):
    self.mask = None
    
  def forward(self, x):
    self.mask = x<=0
    out = x.copy()
    out[self.mask] = 0
    
    return out
  
  def backward(self, dout):
    dout[self.mask] = 0
    dx = dout
    
    return dx

class Sigmoid:
  def __init__(self):
    self.out = None
  
  def forward(self, x):
    out = 1 / (1+np.exp(-x))
    return out
  
  def backward(self, dout):
    dx = dout * (1.0 - self.out) * self.out
    return dx

# https://mathtrain.jp/rensaritsu
# http://w3e.kanazawa-it.ac.jp/math/category/bibun/henbibun/henkan-tex.cgi?target=/math/category/bibun/henbibun/gouseikansuu-no-henbibun_doushutu1.html
# https://qiita.com/sand/items/2d783a12c575fb949c6e
class Affine:
  def __init__(self, W, b):
    self.W = W, self.b = b
    self.x = self.dW = self.db = None
    
  def forward(self, x):
    self.x = x
    out = np.dot(x, self.W) + b
    
    return out
    
  def backward(self, dout):
    # ∂L/∂X = ∂L/∂Y*W^T
    dx = np.dot(dout, self.W.T)
    # ∂L/∂W = X^T*∂L/∂Y
    dw = np.dot(self.X.T, dout)
    # ∂L/∂B = ∑ { ∂L/∂Y_{i,j} : i }
    db = np.sum(dout, axis=0)
    
    return dx

class SoftmaxWithLoss:
  def __init__(self):
    self.loss = None
    self.y = None # ouput of softmax
    self.t = None # training data in one-hot format
  
  def forward(self, x, t):
    self.t = t
    self.y = softmax(x)
    self.loss = cross_entropy_error(self.y, self.t)
    
    return self.loss
  
  def backward(self, dout=1):
    batch_size = self.t.shape[0]
    dx = (self.y - self.t) / batch_size
    
    return dx

def nearlyEqual(x, y):
  EPS=1e-5
  return abs(x-y) < EPS

def showAppleExample():
  apple = 100
  apple_num = 2
  tax = 1.1
  
  # layer
  mul_apple_layer = MultiLayer()
  mul_tax_layer = MultiLayer()
  
  # forward, mul_apple_layer -> mul_tax_layer
  apple_price = mul_apple_layer.forward(apple, apple_num)
  price = mul_tax_layer.forward(apple_price, tax)
  assert nearlyEqual(price, 220), "price = 100*2*1.1"
  
  # backward, mul_tax_layer -> mul_apple_layer
  dprice = 1
  dapple_price, dtax = mul_tax_layer.backward(dprice)
  dapple, dapple_num = mul_apple_layer.backward(dapple_price)
  assert nearlyEqual(dapple, 2.2), "dapple = 1*1.1*2"
  assert nearlyEqual(dapple_num, 110), "dapple = 1*1.1*100"
  assert nearlyEqual(dtax, 200), "dapple = 1*200"

def showAppleOrangeExample():
  apple = 100
  apple_num = 2
  orange = 150
  orange_num = 3
  tax = 1.1
  
  # layer
  mul_apple_layer = MultiLayer()
  mul_orange_layer = MultiLayer()
  add_apple_orange_layer = AddLayer()
  mul_tax_layer = MultiLayer()
  
  # forward
  apple_price = mul_apple_layer.forward(apple, apple_num)
  orange_price = mul_orange_layer.forward(orange, orange_num)
  all_price = add_apple_orange_layer.forward(apple_price, orange_price)
  price = mul_tax_layer.forward(all_price, tax)
  
  assert nearlyEqual(price, 715), "price=(2*100+3*150)*1.1"
  
  # backward
  dprice = 1
  dall_price, dtax = mul_tax_layer.backward(dprice)
  dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
  dorange, dorange_num = mul_orange_layer.backward(dorange_price)
  dapple, dapple_num = mul_apple_layer.backward(dapple_price)
  
  assert nearlyEqual(dtax, 650), "price=1*650"
  assert nearlyEqual(dapple_num, 110), "dapple_num=1*1.1*100"
  assert nearlyEqual(dall_price, 1.1), "dall_price=1*1.1"
  assert nearlyEqual(dapple, 2.2), "dapple=1*1.1*2"
  assert nearlyEqual(dorange_num, 165), "dapple_num=1*1.1*150"
  assert nearlyEqual(dorange, 3.3), "dorange=1*1.1*3"

showAppleExample()
showAppleOrangeExample()
