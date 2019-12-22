import numpy as np

def NAND(x1, x2):
  w = np.array([-0.5, -0.5])
  x = np.array([x1, x2])
  b = 0.7
  val = np.sum(w*x) + b
  return 0 if val<=0 else 1

assert NAND(0,0)==1, "!(0&0) should be 1"
assert NAND(1,0)==1, "!(1&0) should be 1"
assert NAND(0,1)==1, "!(0&1) should be 1"
assert NAND(1,1)==0, "!(1&1) should be 0"
