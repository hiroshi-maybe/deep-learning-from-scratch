import numpy as np
import matplotlib.pylab as plt

def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x))
  
X = np.arange(-5.0, 5.0, 0.1)
Y = sigmoid(X)
plt.plot(X,Y)
plt.ylim(-0.1,1.1)
plt.show()
