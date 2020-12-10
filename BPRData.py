import numpy as np 
import scipy.sparse as sp
import torch.utils.data as data
import pickle

class BPRData(data.Dataset):
	def __init__(self, data):
		super(BPRData, self).__init__()
		self.data = np.array(data)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		user = self.data[idx][0]
		item_i = self.data[idx][1]
		return user, item_i

		