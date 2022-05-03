import os
import json
import gzip
import math
import pandas as pd
from urllib.request import urlopen
import nltk
from nltk.tokenize import RegexpTokenizer
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import time
import tqdm
import torch.optim as optim
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from scipy import stats

nltk.download('punkt')
vader = None
embeddings_index = None



