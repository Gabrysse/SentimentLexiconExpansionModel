import numpy as np
import torch
from matplotlib import pyplot as plt

from dataset.Utilities import read_vader, read_glove
from neural.net_softmax import NetSoftmax
%matplotlib inline

model = NetSoftmax(0, 0)
model.load_state_dict(torch.load("net1.pth"))
model.eval()

vader = read_vader()
glove = read_glove()

err = []
for word in vader.keys():
    if word in glove.keys():
        ground_truth = vader[word]
        predicted = model(torch.tensor(glove[word]).unsqueeze(dim=0)).detach().item()
        err.append(predicted - ground_truth)

err = np.array(err)

n_bins = 10
n, bins, patches = plt.hist(err, n_bins)
plt.show()
