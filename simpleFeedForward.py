#!/usr/bin/python

import numpy as np
import math
import random
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable

from functools import partial
import collections

np.random.seed(0)

# Generate a clean sine wave
def sine(X, signal_freq=60.):
    return np.sin(2 * np.pi * (X) / signal_freq)

# Add uniform noise
def noisy(Y, noise_range=(-0.35, 0.35)):
    noise = np.random.uniform(noise_range[0], noise_range[1], size=Y.shape)
    return Y + noise

# Create a noisy and clean sine wave
def sample(sample_size):
    random_offset = random.randint(0, sample_size)
    X = np.arange(sample_size)
    out = sine(X + random_offset)
    inp = noisy(out)
    return inp, out


# inp, out = sample(100)
# plt.plot(inp, label='Noisy')
# plt.plot(out, label='Denoised')
# plt.legend()
# plt.show()


# Create a dataset
def create_dataset(n_samples=10000, sample_size=100):
    data_inp = np.zeros((n_samples, sample_size))
    data_out = np.zeros((n_samples, sample_size))

    for i in range(n_samples):
        sample_inp, sample_out = sample(sample_size)
        data_inp[i, :] = sample_inp
        data_out[i, :] = sample_out
    return data_inp, data_out


# Split the data into training and testing partitions
data_inp, data_out = create_dataset()
train_inp, train_out = data_inp[:8000], data_out[:8000]
test_inp, test_out = data_inp[8000:], data_out[8000:]

hidden_size = 30

class Model(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(Model, self).__init__()
		self.hidden = nn.Linear(input_size, hidden_size)
		self.predict = nn.Linear(hidden_size, output_size)
		self.tanh = nn.Tanh()
	
	def forward(self, x):
		tanh = self.tanh(self.hidden(x))
		prediction = self.predict(tanh)
		return prediction


model = Model(input_size=1, hidden_size=30, output_size=1)

epochs = 300
learnrate = 1e-2
optimizer = torch.optim.Adam(model.parameters(), lr=learnrate)
loss_func = nn.L1Loss()


for i in range(epochs+1):
	inp = Variable(torch.Tensor(train_inp.reshape((train_inp.shape[0], -1, 1))))
	out = Variable(torch.Tensor(train_out.reshape((train_out.shape[0], -1, 1))))
	pred = model(inp)
	optimizer.zero_grad()
	loss = loss_func(pred, out)
	if i % 20 == 0:
		print('Epoch: {}\tLoss Value: {}'.format(i, loss.item()))
	loss.backward()
	optimizer.step()

test_inp = Variable(torch.Tensor(test_inp.reshape((test_inp.shape[0], -1, 1))))
pred = model(test_inp)
lossTest = loss_func(pred, Variable(torch.Tensor(test_out.reshape((test_inp.shape[0], -1, 1)))))
print('\nLoss on Test Set: '+str(lossTest.item()))
print()

sample_num = 150
plt.plot(pred[sample_num].data.numpy(), label='Predication')
plt.plot(test_out[sample_num], label='Original')
plt.legend()
plt.show()


# loading test.pt data
activations = collections.defaultdict(list)

def save_activation(name, mod, inp, out):
	activations[name].append(out.cpu())

inp = torch.load('test.pt')
inp = Variable(torch.Tensor(inp.reshape((inp.shape[0], -1, 1))))

for name, m in model.named_modules():
	if name == 'predict':
		m.register_forward_hook(partial(save_activation, name))

pred = model(inp)

act = torch.squeeze(activations['predict'][0], 2).detach().numpy()
act = act[:5].transpose()
plt.plot(act)
plt.show()

lossTest = loss_func(pred, Variable(torch.Tensor(inp.reshape((inp.shape[0], -1, 1)))))
print('\nLoss on Test Set test.pt: '+str(lossTest.item()))
print()
