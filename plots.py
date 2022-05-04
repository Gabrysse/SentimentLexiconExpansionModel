import numpy as np
import torch
from matplotlib import pyplot as plt

from dataset.Utilities import read_vader, read_glove
from neural.net_softmax import NetSoftmax

vader = read_vader()
glove = read_glove()

scale_max = 0
scale_min = 0
for word in vader.keys():
    if word in glove.keys():
        scale_max = max(scale_max, vader[word])
        scale_min = min(scale_min, vader[word])

model = NetSoftmax(scale_min, scale_max)
model.load_state_dict(torch.load("net1.pth"))
model.eval()

err = []
for word in vader.keys():
    if word in glove.keys():
        ground_truth = vader[word]
        predicted = model(torch.tensor(glove[word]).unsqueeze(dim=0)).detach().item()
        err.append(predicted - ground_truth)

err = np.array(err)

n_bins = 20
n, bins, patches = plt.hist(err, n_bins)
plt.savefig('foo.png')
