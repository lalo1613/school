import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from DLcourse.RNN import NetLSTM
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from string import ascii_lowercase
# # Following line opens the nltk downloader GUI from which one can download nltk resources.
# from nltk import download; download()
# # The needed recourses are the "stopwords" corpora and "punkt" package.


# setting directories now
dir_input = r"C:\Users\omri_\Downloads\PTB"+"\\"
# dir_input = r"C:\Users\Bengal\Downloads\PTB"+"\\"


def preprocess_sentence(sentence : str):
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    allowed_symbols = set(l for l in ascii_lowercase)
    output_sentence = []
    for word in word_tokenize(sentence):
        word = str(np.char.lower(word))
        tmp_word = ''
        for letter in word:
            if letter in allowed_symbols:
                tmp_word += letter
        word = tmp_word

        if word not in stop_words:
            word = stemmer.stem(word)
            if len(word) > 1:
                output_sentence.append(word)

    return output_sentence


def preprocess_dataset(dataset):
    dataset = dataset.split('\n')
    dataset_fixed_tr = []
    dataset_fixed_te = []
    for i in dataset:
        tmp = preprocess_sentence(i)
        tmp_input = tmp[:-1]
        tmp_input.insert(0,'SOS')
        tmp_output = tmp[1:]
        tmp_output.append('EOS')
        dataset_fixed_tr.append(tmp_input)
        dataset_fixed_te.append(tmp_output)
    return dataset_fixed_tr, dataset_fixed_te


def Vocab(sentence, Vocabulary):
    n = len(Vocabulary)
    for word in sentence:
        if word not in Vocabulary:
            Vocabulary[word] = n
            n += 1
    return Vocabulary


def create_vocabulary_dict():
    Vocabulary_Dict = {'<PAD>': 0, 'SOS': 1, 'EOS': 2}
    for i in train_input:
        Vocabulary_Dict = Vocab(i, Vocabulary_Dict)
    for i in train_output:
        Vocabulary_Dict = Vocab(i, Vocabulary_Dict)
    for i in valid_input:
        Vocabulary_Dict = Vocab(i, Vocabulary_Dict)
    for i in valid_output:
        Vocabulary_Dict = Vocab(i, Vocabulary_Dict)
    for i in test_input:
        Vocabulary_Dict = Vocab(i, Vocabulary_Dict)
    for i in test_output:
        Vocabulary_Dict = Vocab(i, Vocabulary_Dict)
    return Vocabulary_Dict


def replace_words_nums(dataset):
    dataset_numbers = []
    for sentence in dataset:
        sentence_tmp = []
        for word in sentence:
            sentence_tmp.append(Vocabulary_Dict[word])
        dataset_numbers.append(sentence_tmp)
    return dataset_numbers


def format_to_torch(input_list, output_list):
    input_list = replace_words_nums(input_list)
    input_list.sort(key=len,reverse=True)
    temp = pd.DataFrame(input_list).fillna(0)
    torched_input = torch.Tensor(np.array(temp)).long()

    output_list = replace_words_nums(output_list)
    output_list.sort(key=len,reverse=True)
    temp = pd.DataFrame(output_list).fillna(0)
    torched_output = torch.Tensor(np.array(temp)).long()

    data_lengths = torch.Tensor([len(x) for x in input_list])

    return torched_input, torched_output, data_lengths


# load data:
train = open(dir_input+'ptb.train.txt').read()
valid = open(dir_input+'ptb.valid.txt').read()
test = open(dir_input+'ptb.test.txt').read()

# preproccess data:
train_input, train_output = preprocess_dataset(train)
valid_input, valid_output = preprocess_dataset(valid)
test_input, test_output = preprocess_dataset(test)

# creating vocabulary dictionary:
Vocabulary_Dict = create_vocabulary_dict()

train_input_formatted, train_output_formatted, train_lengths = format_to_torch(train_input, train_output)

# train temp
loss_function = torch.nn.functional.cross_entropy
net = NetLSTM(vocabulary=Vocabulary_Dict, hidden_size=200, embd_dim=3)
optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
batch_size = 50
hidden = None

prep_list = []
for epoch in tqdm(range(5)):
    optimizer.zero_grad()
    # getting probs
    probs, hidden = net(inputs=train_input_formatted[:batch_size], input_lengths=train_lengths[:batch_size], hidden=hidden)
    # calculating loss
    loss = loss_function(probs.view(-1, probs.shape[2]), train_output_formatted[:batch_size].view(-1))
    preplexity = loss.exp()
    # updating parameters
    old = list(net.parameters())[0].clone()
    loss.backward(retain_graph=True)
    optimizer.step()
    new = list(net.parameters())[0].clone()
    print("parameters have changed: "+str(not torch.equal(new, old)))
    # saving preplexity
    prep_list.append(preplexity.item())

print(prep_list)
