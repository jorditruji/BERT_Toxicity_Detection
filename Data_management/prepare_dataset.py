# coding=utf-8

from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import random
import sys
import pandas as pd 
import numpy as np
import re
import torch
import pickle as pkl
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

from data_helpers import InputFeatures, InputExample, convert_examples_to_features, read_examples



mode = 'regression'

train, labels, toxicity = read_examples('../../train.csv', output_mode = mode)


#train = pd.read_csv('../../Datasets/kaggle/train.csv', index_col='id')
#test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv', index_col='id')

mode = 'regression'
# Bert tokenizer
maxlen = 84
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case= True)
num_labels = 2

train_features = convert_examples_to_features(train, ["OK", "Toxic"], maxlen, tokenizer,mode )


all_input_ids = [f.input_ids for f in train_features]
all_input_mask = [f.input_mask for f in train_features]
all_segment_ids = [f.segment_ids for f in train_features]

if mode == "classification":
    all_label_ids = [f.label_id for f in train_features]
elif mode == "regression":
    all_label_ids = [f.label_id for f in train_features]

del(train_features)
#train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

data_save = {}
data_save['all_input_ids'] = all_input_ids
data_save['all_input_mask'] = all_input_mask
data_save['all_segment_ids'] = all_segment_ids
data_save['all_label_ids'] = all_label_ids


filename = 'classification_slen84.pkl'
fileObject = open(filename, 'wb')
pkl.dump(data_save, fileObject)
fileObject.close()

 
