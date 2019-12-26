
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

def nearlyEqual(x, y):
  EPS=1e-5
  return abs(x-y) < EPS

def show():
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

show()
