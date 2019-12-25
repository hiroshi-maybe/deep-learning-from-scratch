
def numerical_gradient(f, x):
  h = 1e-4 # 0.0001
  grad = np.zeros_like(x)

  for idx in range(x.size):
    x0 = x[idx]
    x[idx] = x0+h
    fxh1 = f(x) # f(x+h)
  
    x[idx] = x0-h
    fxh2 = f(x)
  
    grad[idx] = (fxh1-fxh2) / (2*h)
    x[idx] = x0

  return grad
