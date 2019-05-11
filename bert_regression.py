# coding=utf-8

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys
import pandas as pd 
import numpy as np
import re
import torch

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

from Data_management.data_helpers import InputFeatures, InputExample, convert_examples_to_features, read_examples
import torch.nn.functional as F

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)




# Regress or classify
mode = 'regression'
#mode = 'classification'

train, labels, toxicity = read_examples('../train.csv', output_mode = mode)
print("Sample")
print(train[0].text_a, train[0].text_b, train[0].target)


#train = pd.read_csv('../../Datasets/kaggle/train.csv', index_col='id')
#test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv', index_col='id')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'
n_gpu = 1
# Bert tokenizer
maxlen = 84
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case= True)
num_labels = 1

train_features = convert_examples_to_features(train, ["OK", "Toxic"], maxlen, tokenizer,mode )


all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)

if mode == "classification":
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
elif mode == "regression":
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.float)

del(train_features)
train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
train_sampler = RandomSampler(train_data)

batch_size = 16
# Parameters of the data loader
params = {'batch_size': batch_size ,
          'sampler': train_sampler,
          'num_workers': 4,
          'pin_memory': True}

train_dataloader = DataLoader(train_data, **params)


model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels= num_labels)
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

num_train_epochs = 4
gradient_accumulation_steps = 1
num_train_optimization_steps = int(len(train) / batch_size ) * num_train_epochs
print(num_train_optimization_steps)

optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=5e-5,
                     warmup=0.1,
                     t_total=num_train_optimization_steps)
global_step = 0
nb_tr_steps = 0
tr_loss = 0
model.train()
model.to(device)
for _ in trange(int(num_train_epochs), desc="Epoch"):
    running_corrects = 0
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        # define a new function to compute loss values for both output_modes
        logits = model(input_ids, segment_ids, input_mask, labels=None)
        if mode == "classification":
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        elif mode == "regression":
            loss_fct = MSELoss()
            # Target tocicity is between 0 and 1
            logits = F.sigmoid(logits)
            loss = loss_fct(logits.view(-1), label_ids.view(-1))
        loss.backward()

        # Select maximum score index
        _, preds = torch.max(logits, 1)

        running_corrects += float(torch.sum(preds.data == label_ids.data))
        tr_loss += loss.item()
        nb_tr_examples += input_ids.size(0)

        nb_tr_steps += 1
        optimizer.step()
        optimizer.zero_grad()
        global_step += 1
        if step%500 == 0:
          print("Accuracy at step {}: {}, LOSS: {}".format( step, 
            running_corrects/nb_tr_examples,float(tr_loss)/nb_tr_examples))
    torch.save(model.state_dict(), 'bert_regression_Epoch_'+str(_))
    epoch_acc = running_corrects.double().detach() / nb_tr_examples
    epoch_acc = epoch_acc.data.cpu().numpy()
    train_loss = tr_loss/nb_tr_examples
    print("Epoch {}, accuracy: {}, loss: {}".format(_, epoch_acc,train_loss ))

