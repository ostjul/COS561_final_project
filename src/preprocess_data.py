import os
import torch
import numpy as np

DATA_DIR = os.path.join('data', 'dqn_data')

'''
from deviceModel.py class deepQueueNet
self.fet_input(): feature extraction input
    * read file from ./data/<model name>/FIFO/_traces/_<train/test>
    * for each file, convert to a csv and save to ./data/<model name>/FIFO/_traces/<train/test> !! no underscore
self._2hdf()
self._min_max()
self.model_input()
self._loop_size()
self.merge_sample(task='train')
self.merge_sample(task='test1')
self.merge_sample(task='test2')
'''