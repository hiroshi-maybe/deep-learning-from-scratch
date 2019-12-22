import numpy as np
import matplotlib.pylab as plt

# Rectified Linear Unit a.k.a. ReLU
def relu(x):
  return np.maximum(0.0, x)

X = np.arange(-5.0, 5.0, 0.1)
Y = relu(X)
plt.plot(X,Y)
plt.ylim(-0.1,1.1)
plt.show()
