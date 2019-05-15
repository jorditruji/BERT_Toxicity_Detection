# coding=utf-8

from __future__ import absolute_import, division, print_function
import logging
import csv
import re
import random
from sklearn.model_selection import StratifiedShuffleSplit 

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, target = None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, target = None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.target = target

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()




def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features

# Load data
def read_examples(input_file, output_mode = 'classification'):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    labels = []
    toxicity = []
    unique_id = 0
    with open(input_file, "r", encoding='utf-8') as reader:
        csv_reader = csv.reader(reader, delimiter=',')
        for _i, line in enumerate(csv_reader):
            if _i:
                target = float(line[1])
                if target >= 0.5:
                	label = "Toxic"
                else:
                	label = "OK"
                text = line[2]
                text_a = None
                text_b = None
                m = re.match(r"^(.*) \|\|\| (.*)$", text)
                if m is None:
                    text_a = text
                else:
                    print(text_a, text_b)
                    text_a = m.group(1)
                    text_b = m.group(2)
                if output_mode != 'classification':
                    label = target
                examples.append(
                    InputExample(guid=unique_id, text_a=text_a, text_b=text_b, label = label))
                labels.append(label)
                toxicity.append(target)
                unique_id += 1
    return examples, labels, toxicity



def read_from_pkl(fname):
    # reads pickled dataset, u can pickle the dataset using prepare_dataset
    with (open(fname, "rb")) as openfile:
        data = pickle.load(openfile)
    return data

def read_splits(fname, train_size = 0.3, random_state = 1993):
    # Reads pickled data and returns train_test splits
    data = read_from_pkl(fname)
    labels = np.array(data['all_label_ids'],dtype= int)
    # 5 Toxicity bins
    thresholds = [(0,0.2), (0.2,0.4) , (0.4,0.6), (0.6,0.8) ,(0,2 )]
    idx_train = []
    idx_test = []
    for low,high in thresholds:
        # Filter bin
        partition = low<=labels<high
        partition_idx = np.where(partition)
        # Calculate how many samples we need for each partition
        n_train_samples = int(train_size*len(partition_idx))

        # Get train indices
        train_part_idx = random.choice(partition_idx, n_train_samples)
        idx_train+=train_part_idx

        # Get test indices
        idx_test+=partition_idx[partition_idx!=idx_train]
    

    print("Found {} samples. {} toxic comments and {} no toxics".format(n_samples,n_toxics, n_OK))
    



    return x_train, y_train,x_test, y_test