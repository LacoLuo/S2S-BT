'''
It is a modified code offered at https://github.com/malrabeiah/VABT/blob/master/data_feed.py
---------------
Author: Hao Luo
Sep. 2021
'''

import numpy as np 
import pandas as pd 
import random
from skimage import io
from PIL import Image
import jpeg4py as jpeg
import multiprocessing

import torch 
from torch.utils.data import Dataset


def create_samples(root, shuffle=False, nat_sort=False):
	f = pd.read_csv(root)
	f = f.replace(to_replace= r'\\', value= '/', regex=True)
	data_samples = []
	for idx, row in f.iterrows():
		beams = row.values[0:13].astype(np.float32)
		sample = list(beams)
		data_samples.append(sample)

	if shuffle:
		random.shuffle(data_samples)
	return data_samples

class DataFeed(Dataset):
	"""
	A class fetching a PyTorch tensor of beam indices.
	"""

	def __init__(self, root_dir,
				n,
				init_shuffle=True):
		
		self.root = root_dir
		self.samples = create_samples(self.root, shuffle=init_shuffle)
		self.seq_len = n

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		sample = self.samples[idx] # Read one data sample
		
		sample = sample[:self.seq_len] # Read a sequence of tuples from a sample
		beams = torch.zeros((self.seq_len,))

		for i,s in enumerate( sample ):
			beams[i] = torch.tensor(s-1)
        
		return beams
