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

def XOR(x1, x2):
  a = NAND(x1, x2)
  b = OR(x1, x2)
  return AND(a, b)

assert XOR(0,0)==0, "0^0 should be 0"
assert XOR(1,0)==1, "1^0 should be 1"
assert XOR(0,1)==1, "0^1 should be 1"
assert XOR(1,1)==0, "1^1 should be 0"
