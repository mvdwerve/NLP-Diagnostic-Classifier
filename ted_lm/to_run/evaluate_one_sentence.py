# coding: utf-8

import argparse
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import pickle
import os

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--model', type=str,  default='model.pt', help='model to use')
parser.add_argument('--dict', type=str,  default='dict', help='word map to use')
parser.add_argument('--bptt', type=int, default=60, help='sequence length')
args = parser.parse_args()

# function to transform sentence into word id's and put them in a 
# pytorch Variable
# NB Assumes the sentence is already tokenised!
def tokenise(sentence, dictionary):
    words = sentence.split(' ')
    l = len(words)
    assert l <= args.bptt, "sentence too long"
    token = 0
    ids = torch.LongTensor(l)

    for word in words:
        try:
            ids[token] = dictionary.word2idx[word]
        except KeyError:
            print "%s unknown, replace by <unk>" % word
            raw_input()
            ids[token] = dictionary.word2idx['<unk>']
        token += 1
    return ids

# softmax function
softmax = nn.Softmax()

def evaluate(model, dictionary, sentence):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    ntokens = dictionary.ntokens()
    hidden = model.init_hidden(1)

    test_data = tokenise(test_sentence, dictionary)
    input_data = Variable(test_data, volatile=False)

    output, hidden = model(input_data, hidden)
    output_flat = output.view(-1, ntokens)
    logit = output[-1, :]
    sm = softmax(logit).view(ntokens)
    
    def get_prob(word):
        return sm[dictionary.word2idx[word]].data[0]

    print '\n'.join(
            ['%s: %f' % (word, get_prob(word)) for word in check_words]
            )

    return

if __name__ == '__main__':
    # test sentence and words to check
    test_sentence = 'this is a sentence with seven'
    check_words = ['words', 'characters', 'Thursday', 'days', 'walk']
    print test_sentence, '\n'

    # Load dictionary with word ids, load model
    dictionary = pickle.load(open(args.dict, 'rb'))
    with open(args.model, 'rb') as f:
        model = torch.load(f, map_location=lambda storage, loc: storage)

    evaluate(model, dictionary, test_sentence)
