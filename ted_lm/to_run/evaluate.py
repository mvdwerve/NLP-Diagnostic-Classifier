# coding: utf-8

import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import math

import data
import model
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='location of the test data')
parser.add_argument('--model', type=str, help='path to load the model')
parser.add_argument('--dict', type=str, help='path to load the model dict')
parser.add_argument('--bptt', type=int, default=60, help='sequence length')
parser.add_argument('--eval_batch_size', type=int, default=20, help='eval batch size')
parser.add_argument('--cuda', action='store_true')

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

def get_batch(source, i, evaluation=False):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    target = Variable(source[i+1:i+1+seq_len].view(-1))
    return data, target


def evaluate(model, dictionary, data_source, criterion, eval_batch_size):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = dictionary.ntokens()
    hidden = model.init_hidden(eval_batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, evaluation=True)
        output, hidden = model(data, hidden)
        output_flat = output.view(-1, ntokens)
        total_loss += len(data) * criterion(output_flat, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss[0] / len(data_source)


args = parser.parse_args()
dictionary = pickle.load(open(args.dict, 'rb'))

with open(args.model, 'rb') as f:
    model = torch.load(f, map_location=lambda storage, loc: storage)

criterion = nn.CrossEntropyLoss()

corpus_test = dictionary.tokenise_corpus(args.data)
test_data = batchify(corpus_test, args.eval_batch_size)
test_loss = evaluate(model, dictionary, test_data, criterion, args.eval_batch_size)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
