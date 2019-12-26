
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
