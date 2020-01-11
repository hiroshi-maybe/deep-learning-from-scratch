# coding: utf-8
import os
import sys
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net import MultiLayerNet
from common.optimizer import SGD

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

x_train = x_train[:300]
t_train = t_train[:300]

weight_decay_lambda = 0 # No weight decay
#weight_decay_lambda = 0.1

network1 = MultiLayerNet(input_size=784, hidden_size_list=[100,100,100,100,100,100], output_size=10, weight_decay_lambda=0)
network2 = MultiLayerNet(input_size=784, hidden_size_list=[100,100,100,100,100,100], output_size=10, weight_decay_lambda=0.1)

optimizer = SGD(lr=0.01)

max_epochs = 201
train_size = x_train.shape[0]
batch_size = 100

train1_acc_list=[]
train2_acc_list=[]
test1_acc_list=[]
test2_acc_list=[]

iter_per_epoch = max(train_size / batch_size, 1)
epoch_cnt=0

for i in range(1000000000):
  batch_mask = np.random.choice(train_size, batch_size)
  x_batch = x_train[batch_mask]
  t_batch = t_train[batch_mask]

  grads1 = network1.gradient(x_batch, t_batch)
  optimizer.update(network1.params, grads1)
  grads2 = network2.gradient(x_batch, t_batch)
  optimizer.update(network2.params, grads2)

  if i%iter_per_epoch == 0:
    train_acc1 = network1.accuracy(x_train, t_train)
    test_acc1 = network1.accuracy(x_test, t_test)
    train1_acc_list.append(train_acc1)
    test1_acc_list.append(test_acc1)
    train_acc2 = network2.accuracy(x_train, t_train)
    test_acc2 = network2.accuracy(x_test, t_test)
    train2_acc_list.append(train_acc2)
    test2_acc_list.append(test_acc2)

    print("epoch:" + str(epoch_cnt) + ", train acc 1:" + str(train_acc1) + ", test acc 1:" + str(test_acc1) + ", train acc 2:" + str(train_acc2) + ", test acc 2:" + str(test_acc2))

    epoch_cnt += 1
    if epoch_cnt >= max_epochs:
        break

markers = {'train without decay': 'o', 'test without decay': 's', 'train with decay': 'o', 'test with decay': 's'}
x = np.arange(max_epochs)
plt.plot(x, train1_acc_list, marker='o', label='train without decay', markevery=10)
plt.plot(x, test1_acc_list, marker='s', label='test without decay', markevery=10)
plt.plot(x, train2_acc_list, marker='o', label='train with decay', markevery=10)
plt.plot(x, test2_acc_list, marker='s', label='test with decay', markevery=10)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
