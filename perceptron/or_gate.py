import numpy as np

def OR(x1, x2):
  w = np.array([0.5, 0.5])
  x = np.array([x1, x2])
  b = 0.0
  val = np.sum(w*x) + b
  return 0 if val<=0 else 1

assert OR(0,0)==0, "0|0 should be 0"
assert OR(1,0)==1, "1|0 should be 1"
assert OR(0,1)==1, "0|1 should be 1"
assert OR(1,1)==1, "1|1 should be 1"
