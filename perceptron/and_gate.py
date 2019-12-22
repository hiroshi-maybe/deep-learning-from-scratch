import numpy as np

def AND(x1, x2):
  w = np.array([0.5, 0.5])
  x = np.array([x1, x2])
  b = -0.7
  val = np.sum(w*x) + b
  return 0 if val<=0 else 1

assert AND(0,0)==0, "0&0 should be 0"
assert AND(1,0)==0, "1&0 should be 0"
assert AND(0,1)==0, "0&1 should be 0"
assert AND(1,1)==1, "1&1 should be 1"
