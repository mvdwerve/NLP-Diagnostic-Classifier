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
args = parser.parse_args()

softmax = nn.Softmax()
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

print(target_words)
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

net = Net()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
ce_loss = nn.CrossEntropyLoss()
# function to transform sentence into word id's and put them in a pytorch Variable
# NB Assumes the sentence is already tokenised!
def tokenise(sentence, dictionary):
    words = sentence.split(' ')
    l = len(words)
    assert l <= max_seq_len, "Sentence too long"
    token = 0
    ids = torch.LongTensor(l)

    for word in words:
        try:
            ids[token] = dictionary.word2idx[word]
        except KeyError:
            print('Unknown token')
            ids[token] = dictionary.word2idx['<unk>']
        token += 1
    return ids

def evaluate(model, dictionary, sentence, index):
    # Turn on evaluation mode which disables dropout.
    model.eval()

    # number of tokens (= output size)
    ntokens = len(dictionary)
    hidden = model.init_hidden(1)

    # tokenise the sentence, put in torch Variable
    test_data = tokenise(sentence, dictionary)
    input_data = Variable(test_data)

    optimizer.zero_grad()

    # Run the model, compute probabilities by applying softmax
    output, hidden = model(input_data, hidden)

    prediction = net(hidden[0])[1]
    temp = Variable(torch.LongTensor(1))
    target = pos_tags.get(pos_dict.get(target_words[index]))
    if target == None:
        print('Skipped sentence')
        return output, hidden
    temp[0] = target
    print('Target', temp)
    loss = ce_loss(prediction, temp)
    loss.backward()
    optimizer.step()

    output_flat = output.view(-1, ntokens)
    logits = output[-1, :]
    sm = softmax(logits).view(ntokens)

    def get_prob(word):
        try:
            return sm[dictionary.word2idx[word]].data[0]
        except KeyError:
            return 0

    # print('\n'.join(
    #     ['%s: %f' % (word, get_prob(word)) for word in check_words]
    # ))

    return output, hidden
#
def plot_tensor(tensors, indices):
    plt.imshow(tensors[:, indices])
    plt.colorbar()
    plt.show()

for index, sentence in enumerate(sentences):
    print(index, '/', len(sentences))
    output, hidden = evaluate(lm, dictionary, sentence, index)
    # Visualize hidden layer
    # indices = [dictionary.word2idx[word] for word in check_words]
    # img_data = output.view(-1, len(dictionary)).data.numpy()

torch.save(net, 'diagnostic-classifier.pt')

print('Evaluating diagnostic classifier...')

for index, sentence in enumerate(sentences[:10]):
    _, hidden = evaluate(lm, dictionary, sentence, index)
    prediction = net(hidden[0]).data.numpy().tolist()[0]
    print('Word to predict:', target_words[index])
    print('Training sentence:', sentence)
    print('Predicted POS tag:', get_pos_tag(prediction[0].index(max(prediction[0]))))
    print('-------------------------')

# print('Tensor dimensions:', img_data.shape)

# plot_tensor(img_data, indices)