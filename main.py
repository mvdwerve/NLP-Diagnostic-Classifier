import argparse

import torch
from torch.autograd import Variable
import torch.nn as nn
import pickle
import numpy as np
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import optim

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size (default: 32)')
args = parser.parse_args()

# make the dictionaries, tags and words
pos_dict = {}
pos_tags = {}
sentences = []
target_words = []

with open('ted_dict_words_pos.txt') as f:
    content = f.readlines()

    for line in content:
        line = line.strip()
        words = line.split(' ')
        for word in words:
            tagged_word = word.split('/')
            if len(tagged_word) > 1:
                if pos_tags.get(tagged_word[1]) == None:
                    pos_tags[tagged_word[1]] = len(pos_tags)
                pos_dict[tagged_word[0]] = tagged_word[1]

def get_pos_tag(index):
    return [k for (k, v) in list(pos_tags.items()) if v == index]

with open('ted_lm/to_run/data/train.txt') as f:
    content = f.readlines()

    for line in content:
        if len(line.split(' ')) > 2:
            sentences.append(' '.join(line.split(' ')[:-2]))
            target_words.append(line.split(' ')[-2])

# load the model and dictionary into memory
lm = torch.load('ted_lm/to_run/model-lr-10-min5.pt', map_location=lambda storage, loc: storage)
dictionary = pickle.load(open('ted_lm/to_run/ted_min5.dict', 'rb'), encoding='utf8')
max_seq_len = 50

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1500, 500)
        self.fc2 = nn.Linear(500, 45)

    def forward(self, x):
        x = F.selu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x)

# net = torch.load('diagnostic-classifier.pt', map_location=lambda storage, loc: storage)

class Trainer:
    def __init__(self, model, dictionary):
        # initialize vars for training the net
        self.net = Net()
        self.optimizer =  optim.SGD(self.net.parameters(), lr=args.lr, momentum=args.momentum)
        self.loss = nn.CrossEntropyLoss()
       
        # store the model and some utility variables
        self.model = model
        self.dictionary = dictionary
        self.ntokens = len(dictionary)
        self.hidden = self.model.init_hidden(1)

        # Turn on evaluation mode which disables dropout.
        self.model.eval()
    
    def get_id(self, word):
        return dictionary.word2idx[word] if word in self.dictionary.word2idx else self.dictionary.word2idx['<unk>']

    # function to transform sentence into word id's and put them in a pytorch Variable
    # NB Assumes the sentence is already tokenised!
    def tokenise(self, sentence):
        words = sentence.split(' ')
        assert len(words) <= max_seq_len, "Sentence too long"
        return torch.LongTensor([self.get_id(w) for w in words])
    
    def evaluate(self, sentences, targets):
        # storing the output here
        outputs, hiddens = [], []

        # make the optimizer zero
        for sentence, target in zip(sentences, targets):
            # set optimizer to zero & get the input for it
            self.optimizer.zero_grad()
            input_data = Variable(self.tokenise(sentence))

            # Run the model, compute probabilities by applying softmax
            output, hidden = self.model(input_data, self.hidden)

            # the prediction in the hidden layer
            prediction = self.net(hidden[0])[1]
            target_id = pos_tags.get(pos_dict.get(target))
            
            # append to the output and hiddens (faux batching)
            outputs.append(output)
            hiddens.append(hidden)
    
            # some user output
            print('%s => %s' % (sentence, target))
            if target_id == None:
                print(" ---> skipped last sentence, because target \"%s\"not in dictionary" % target)
                continue

            # make the loss and optimizer step
            loss = self.loss(prediction, Variable(torch.LongTensor([target_id])))
            loss.backward()
            self.optimizer.step()
        
        # return the output
        return outputs, hiddens

def plot_tensor(tensors, indices):
    plt.imshow(tensors[:, indices])
    plt.colorbar()
    plt.show()

# create a trainer and set batch size
trainer = Trainer(lm, dictionary)

# loop over the batches
for i in range(0, len(sentences), args.batch_size):
    print(i + 1, '/', len(sentences))
    outputs, hiddens = trainer.evaluate(sentences[i:i+args.batch_size], target_words[i:i+args.batch_size])

# save our own neural net
torch.save(net, 'diagnostic-classifier.pt')

print('Evaluating diagnostic classifier...')

for i in range(0, len(sentences), batch_size):
    _, hiddens = trainer.evaluate(sentences[i:i+args.batch_size], target_words[i:i+args.batch_size])
    prediction = trainer.net(trainer.hidden[0]).data.numpy().tolist()[0]
    print('Word to predict:', target_words[index])
    print('Training sentence:', sentence)
    print('Predicted POS tag:', get_pos_tag(prediction[0].index(max(prediction[0]))))
    print('-------------------------')
