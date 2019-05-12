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





train, labels, toxicity = read_examples('../../train.csv')


#train = pd.read_csv('../../Datasets/kaggle/train.csv', index_col='id')
#test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv', index_col='id')

mode = 'classification'
# Bert tokenizer
maxlen = 84
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case= True)
num_labels = 2

train_features = convert_examples_to_features(train, ["OK", "Toxic"], maxlen, tokenizer,mode )


all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)

if mode == "classification":
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
elif mode == "regression":
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.float)

del(train_features)
#train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

data_save = [all_input_ids.numpy(), all_input_mask.numpy(), all_segment_ids.numpy(), all_label_ids.numpy()]

filename = 'classification_slen84'
fileObject = open(fileName, 'wb')


pkl.dump(data_save, fileObject)
fileObject.close()

 