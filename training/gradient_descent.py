import numpy as np

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

def function_2(x):
  return x[0]**2 + x[1]**2

def gradient_descent(f, init_x, lr=0.01, step_num=100):
  x = init_x
  
  for i in range(step_num):
    grad = numerical_gradient(f, x)
    x -= lr*grad
    
  return x

init_x = np.array([-3.0, 4.0])
res = gradient_descent(function_2, init_x = init_x, lr=0.1, step_num=100)
print(res) # Right minimal obtained

res = gradient_descent(function_2, init_x = init_x, lr=10.0, step_num=100)
print(res) # Too large learning rate

res = gradient_descent(function_2, init_x = init_x, lr=1e-10, step_num=100)
print(res) # Too small learning rate
