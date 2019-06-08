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

from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

from Data_management.data_helpers import InputFeatures, InputExample, convert_examples_to_features, read_examples, read_from_pkl
import torch.nn.functional as F

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


# weights_loader
def load_weights_sequential(target, source_state):
    model_to_load= {k: v for k, v in source_state.items() if k in target.state_dict().keys()}
    target.load_state_dict(model_to_load)



# Regress or classify
mode = 'regression'

# Read dataset and train vs test indices
data = read_from_pkl('Data_management/classification_slen84.pkl')
idx_partitions = np.load('Data_management/partition_idx.npy').item()

# Convert dataset to tensors
all_label_ids = torch.tensor(data['all_label_ids'], dtype=torch.float)
all_input_ids = torch.tensor(data['all_input_ids'], dtype= torch.long)
all_input_mask =  torch.tensor(data['all_input_mask'], dtype= torch.long)
all_segment_ids =  torch.tensor(data['all_segment_ids'], dtype= torch.long)
all_weights =  torch.tensor(data['all_weights'], dtype= torch.long)


train_data = TensorDataset(all_input_ids[idx_partitions['val']], all_input_mask[idx_partitions['val']], all_segment_ids[idx_partitions['val']], all_label_ids[idx_partitions['val']])
test_sampler = RandomSampler(train_data)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 64
# Parameters of the data loader
params = {'batch_size': batch_size ,
         # 'sampler': train_sampler,
          'num_workers': 6,
          'pin_memory': True}

train_dataloader = DataLoader(train_data, **params)

num_labels= 1

# Load weights
weights_path = 'bert_regression_Epoch_3'
trained = torch.load(weights_path,map_location='cpu')

# Delete state_dict = trained if u want to take original weights
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels= num_labels, state_dict = trained)
print(model)

if mode == "regression":
    #Class weights
    pos_weight = torch.tensor([1.5]).to(device)
    loss_fct = MSELoss()#BCEWithLogitsLoss()#pos_weight=pos_weight)



num_train_epochs = 1
gradient_accumulation_steps = 1
num_train_optimization_steps = int(len(train_data) / batch_size ) * num_train_epochs


global_step = 0
nb_tr_steps = 0
tr_loss = 0
model.eval()
history = []
model.to(device)
for _ in trange(int(num_train_epochs), desc="Epoch"):
    running_corrects = 0.
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        # define a new function to compute loss values for both output_modes
        logits = model(input_ids, segment_ids, input_mask, labels=None)
        if mode == "classification":
            loss = loss_fct(logits, label_ids)
        elif mode == "regression":
            # Weights
            #loss_fct = MSELoss()
            # Target tocicity is between 0 and 1
            logits = F.sigmoid(logits)
            if step%250 == 0:
                print( logits.view(-1))
                print(label_ids)
            loss = loss_fct(10*label_ids,10*logits.view(-1) )
            #Boolean torchie tensor for toxics vs no tocisx
            #logits = F.sigmoid(logits)
            #print("Diff:",logits.view(-1)-label_ids)
            preds = logits.view(-1) >= 0.5
            #print(logits,preds)
            ground_truth = label_ids >= 0.5
            #print(label_ids, ground_truth)
            running_corrects += torch.sum(ground_truth==preds)



            #print(running_corrects, ground_truth == preds)
        # Track losses, amont of samples and amount of gradient steps
        tr_loss +=  loss.item()#*input_ids.size(0)
        nb_tr_examples += input_ids.size(0)
        #print(float(running_corrects), nb_tr_examples)
        nb_tr_steps += 1


    epoch_acc = running_corrects.double().detach() / nb_tr_examples
    epoch_acc = epoch_acc.data.cpu().numpy()
    train_loss = tr_loss/nb_tr_steps
    print("Epoch {}, accuracy: {}, loss: {}".format(_, epoch_acc,train_loss ))
    history.append([train_loss,epoch_acc])
