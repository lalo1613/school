import gzip
import os
import re
import shutil
import pandas as pd
from mnist import MNIST
import numpy as np
import torch
from DLcourse.lenet5 import Net_BatchNorm, Net_Dropout, Net_Dropout_BatchNorm, Net_None
from DLcourse.training import Training_LENET
from DLcourse.testing import TestingNet
import matplotlib.pyplot as plt



# setting directories now
dir_input = r"C:\Users\Bengal\Downloads\PTB"+"\\"

#uncompressing image files and moving them to separate directory (run once only!)

file_list = os.listdir(dir_input)

from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import nltk
from nltk.corpus import stopwords
from string import punctuation, ascii_lowercase


stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))
allowed_symbols = set(l for l in ascii_lowercase)

def preprocess_sentence(sentence : str):
    output_sentence = []
    for word in word_tokenize(sentence):
        word = str(np.char.lower(word))
        tmp_word = ''
        for letter in word:
            if letter in allowed_symbols:
                tmp_word += letter
        word = tmp_word

        if (word in stop_words):
            next
        else:
            word = stemmer.stem(word)
            if len(word) > 1:
                output_sentence.append(word)
    return output_sentence

# loading train and test data
train = open(dir_input+'ptb.train.txt').read()
valid = open(dir_input+'ptb.valid.txt').read()
test = open(dir_input+'ptb.test.txt').read()

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

train_input,train_output = preprocess_dataset(train)
valid_input,valid_output = preprocess_dataset(valid)
test_input,test_output = preprocess_dataset(test)

##################################################
#   CREATE A DICTIONARY

def Vocab(sentence, Vocabulary):
    n = len(Vocabulary)
    for word in sentence:
        if word in Vocabulary:
            next;
        else:
            Vocabulary[word] = n
            n += 1
    return Vocabulary

Vocabulary_Dict = {'SOS':0,'EOS':1}
for i in train_input:
    Vocabulary_Dict = Vocab(i,Vocabulary_Dict)
for i in train_output:
    Vocabulary_Dict = Vocab(i,Vocabulary_Dict)
for i in valid_input:
    print(i)
    Vocabulary_Dict = Vocab(i,Vocabulary_Dict)
for i in valid_output:
    Vocabulary_Dict = Vocab(i,Vocabulary_Dict)
for i in test_input:
    Vocabulary_Dict = Vocab(i,Vocabulary_Dict)
for i in test_output:
    Vocabulary_Dict = Vocab(i,Vocabulary_Dict)
##################################################

# אני פה עכשיו!!
#   REPLACE WORDS WITH DICTIONARY KEYS
def replace_words_nums(dataset):
    dataset_numbers = []
    for sentence in dataset:
        sentence_tmp = []
        for word in sentence:
            sentence_tmp.append(Vocabulary_Dict[word])
        dataset_numbers.append(sentence_tmp)
    return dataset_numbers

train_input_numbers = replace_words_nums(train_input)
train_output_numbers = replace_words_nums(train_output)
valid_input_numbers = replace_words_nums(valid_input)
valid_output_numbers = replace_words_nums(valid_output)
test_input_numbers = replace_words_nums(test_input)
test_output_numbers = replace_words_nums(test_output)



# saving train and test data as torch tensors
train_input_numbers = torch.FloatTensor(train_input_numbers).float()
train_output_numbers = torch.tensor(train_output_numbers).float()
test_input_numbers = torch.tensor(test_input_numbers).float()
test_output_numbers = torch.tensor(test_output_numbers)


#  LSTM
percp_train, percp_val = Training_LENET(train_input = train_input_numbers, train_output = train_output_numbers,test_input = valid_input_numbers, test_output = valid_output_numbers, dir_input = dir_input ,NetName = Net_LSTM, n_epochs = 1)
acc_l2, acc_l2_tst = Training_LENET(train_images = train_images, train_labels = train_labels,test_images = test_images, test_labels = test_labels, dir_input = dir_input ,NetName = Net_None,optimizer_input= 'l2', n_epochs = 15)

# Dropout
acc_dropout, acc_dropout_tst = Training_LENET(train_images = train_images, train_labels = train_labels,test_images = test_images, test_labels = test_labels, dir_input = dir_input ,NetName = Net_Dropout,optimizer_input= None, n_epochs = 15)
acc_dropout_l2,acc_dropout_l2_tst = Training_LENET(train_images = train_images, train_labels = train_labels, test_images = test_images, test_labels = test_labels,dir_input = dir_input ,NetName = Net_Dropout,optimizer_input= 'l2', n_epochs = 15)

# BatchNorm
acc_batch_norm,acc_batch_norm_tst = Training_LENET(train_images = train_images, train_labels = train_labels,test_images = test_images, test_labels = test_labels, dir_input = dir_input ,NetName = Net_BatchNorm,optimizer_input= None, n_epochs = 15)
acc_batch_norm_l2,acc_batch_norm_l2_tst = Training_LENET(train_images = train_images, train_labels = train_labels,test_images = test_images, test_labels = test_labels, dir_input = dir_input ,NetName = Net_BatchNorm,optimizer_input= 'l2', n_epochs = 15)

# Dropout + BatchNorm
acc_dropout_batch_norm,acc_dropout_batch_norm_tst = Training_LENET(train_images = train_images, train_labels = train_labels,test_images = test_images, test_labels = test_labels, dir_input = dir_input ,NetName = Net_Dropout_BatchNorm,optimizer_input= None, n_epochs = 15)
acc_dropout_batch_norm_l2,acc_dropout_batch_norm_l2_tst = Training_LENET(train_images = train_images, train_labels = train_labels,test_images = test_images, test_labels = test_labels, dir_input = dir_input ,NetName = Net_Dropout_BatchNorm,optimizer_input= 'l2', n_epochs = 15)


# Plots

plt.plot('epoch', 'accuracy', data=acc_none, color='blue', markersize=12, linewidth=4, label= 'None')
plt.plot('epoch', 'accuracy', data=acc_l2, color='red', markersize=12,linewidth=4,  label = 'L2')
plt.plot('epoch', 'accuracy', data=acc_none_tst, color='blue', markersize=12, linewidth=4,linestyle='dashed', label= 'None Test')
plt.plot('epoch', 'accuracy', data=acc_l2_tst, color='red', markersize=12,linewidth=4, linestyle='dashed', label = 'L2 Test')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.title('Training Accuracy Per Epoch')
plt.legend()


plt.plot('epoch', 'accuracy', data=acc_dropout, color='blue', markersize=12, linewidth=4, label = 'Dropout')
plt.plot('epoch', 'accuracy', data=acc_dropout_l2, color='red', markersize=12, linewidth=4,  label = 'Dropout + L2')
plt.plot('epoch', 'accuracy', data=acc_dropout_tst, color='blue', markersize=12, linewidth=4,linestyle='dashed', label = 'Dropout Test')
plt.plot('epoch', 'accuracy', data=acc_dropout_l2_tst, color='red', markersize=12, linewidth=4, linestyle='dashed', label = 'Dropout + L2 Test')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.title('Training Accuracy Per Epoch')
plt.legend()


plt.plot('epoch', 'accuracy', data=acc_dropout_batch_norm, color='blue', markersize=12, linewidth=4, label= 'Dropout + BatchNorm')
plt.plot('epoch', 'accuracy', data=acc_dropout_batch_norm_l2, color='red', markersize=12, linewidth=4,  label= 'Dropout + BatchNorm + L2')
plt.plot('epoch', 'accuracy', data=acc_dropout_batch_norm_tst, color='blue', markersize=12, linewidth=4,linestyle='dashed', label = 'Dropout + BatchNorm Test')
plt.plot('epoch', 'accuracy', data=acc_dropout_batch_norm_l2_tst, color='red', markersize=12, linewidth=4, linestyle='dashed', label = 'Dropout + BatchNorm + L2 Test')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.title('Training Accuracy Per Epoch')
plt.legend()


plt.plot('epoch', 'accuracy', data=acc_batch_norm, color='blue', markersize=12,linewidth=4, label = 'BatchNorm')
plt.plot('epoch', 'accuracy', data=acc_batch_norm_l2, color='red', markersize=12, linewidth=4,  label= 'BatchNorm + L2')
plt.plot('epoch', 'accuracy', data=acc_batch_norm_tst, color='blue', markersize=12,linewidth=4,linestyle='dashed', label ='BatchNorm Test')
plt.plot('epoch', 'accuracy', data=acc_batch_norm_l2_tst, color='red', markersize=12, linewidth=4, linestyle='dashed', label= 'BatchNorm + L2 Test')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.title('Training Accuracy Per Epoch')
plt.legend()

#All Train Together
plt.plot('epoch', 'accuracy', data=acc_none, color='blue', markersize=12, linewidth=4, label= 'None')
plt.plot('epoch', 'accuracy', data=acc_l2, color='blue', markersize=12,linewidth=4, linestyle='dashed', label = 'L2')
plt.plot('epoch', 'accuracy', data=acc_dropout, color='green', markersize=12, linewidth=4, label = 'Dropout')
plt.plot('epoch', 'accuracy', data=acc_dropout_l2, color='green', markersize=12, linewidth=4, linestyle='dashed', label = 'Dropout + L2')
plt.plot('epoch', 'accuracy', data=acc_dropout_batch_norm, color='red', markersize=12, linewidth=4, label = 'Dropout + BatchNorm')
plt.plot('epoch', 'accuracy', data=acc_dropout_batch_norm_l2, color='red', markersize=12, linewidth=4, linestyle='dashed', label = 'Dropout + BatchNorm + L2')
plt.plot('epoch', 'accuracy', data=acc_batch_norm, color='brown', markersize=12,linewidth=4, label = 'BatchNorm')
plt.plot('epoch', 'accuracy', data=acc_batch_norm_l2, color='brown', markersize=12, linewidth=4, linestyle='dashed', label= 'BatchNorm + L2')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.title('Training Accuracy Per Epoch')
plt.legend()

#All Test Together
plt.plot('epoch', 'accuracy', data=acc_none_tst, color='blue', markersize=12, linewidth=4, label= 'None')
plt.plot('epoch', 'accuracy', data=acc_l2_tst, color='blue', markersize=12,linewidth=4, linestyle='dashed', label = 'L2')
plt.plot('epoch', 'accuracy', data=acc_dropout_tst, color='green', markersize=12, linewidth=4, label = 'Dropout')
plt.plot('epoch', 'accuracy', data=acc_dropout_l2_tst, color='green', markersize=12, linewidth=4, linestyle='dashed', label = 'Dropout + L2')
plt.plot('epoch', 'accuracy', data=acc_dropout_batch_norm_tst, color='red', markersize=12, linewidth=4, label = 'Dropout + BatchNorm')
plt.plot('epoch', 'accuracy', data=acc_dropout_batch_norm_l2_tst, color='red', markersize=12, linewidth=4, linestyle='dashed', label = 'Dropout + BatchNorm + L2')
plt.plot('epoch', 'accuracy', data=acc_batch_norm_tst, color='brown', markersize=12,linewidth=4, label = 'BatchNorm')
plt.plot('epoch', 'accuracy', data=acc_batch_norm_l2_tst, color='brown', markersize=12, linewidth=4, linestyle='dashed', label= 'BatchNorm + L2')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.title('Testing Accuracy Per Epoch')
plt.legend()



# acc_none_tst = TestingNet(test_images = test_images, test_labels = test_labels, NetName = Net_None, dir_input = dir_input)
# acc_l2_tst = TestingNet(test_images = test_images, test_labels = test_labels, NetName = Net_None, dir_input = dir_input, optimizer_input = 'l2')
# acc_dropout_tst = TestingNet(test_images = test_images, test_labels = test_labels, NetName = Net_Dropout, dir_input = dir_input)
# acc_dropout_l2_tst = TestingNet(test_images = test_images, test_labels = test_labels, NetName = Net_Dropout, dir_input = dir_input, optimizer_input = 'l2')
# acc_dropout_batch_norm_tst = TestingNet(test_images = test_images, test_labels = test_labels, NetName = Net_Dropout_BatchNorm, dir_input = dir_input)
# acc_dropout_batch_norm_l2_tst = TestingNet(test_images = test_images, test_labels = test_labels, NetName = Net_Dropout_BatchNorm, dir_input = dir_input, optimizer_input = 'l2')
# acc_batch_norm_tst = TestingNet(test_images = test_images, test_labels = test_labels, NetName = Net_BatchNorm, dir_input = dir_input)
# acc_batch_norm_l2_tst = TestingNet(test_images = test_images, test_labels = test_labels, NetName = Net_BatchNorm, dir_input = dir_input, optimizer_input = 'l2')

###############################################################

train_acc_list = [acc_none.loc[14,'accuracy'],acc_l2.loc[14,'accuracy'],acc_dropout.loc[14,'accuracy'],
                  acc_dropout_l2.loc[14,'accuracy'],acc_dropout_batch_norm.loc[14,'accuracy'],
                  acc_dropout_batch_norm_l2.loc[14,'accuracy'], acc_batch_norm.loc[14,'accuracy'],
                  acc_batch_norm_l2.loc[14,'accuracy']]

test_acc_list = [acc_none_tst.loc[14,'accuracy'],acc_l2_tst.loc[14,'accuracy'],acc_dropout_tst.loc[14,'accuracy'],
                 acc_dropout_l2_tst.loc[14,'accuracy'],acc_dropout_batch_norm_tst.loc[14,'accuracy'],
                 acc_dropout_batch_norm_l2_tst.loc[14,'accuracy'], acc_batch_norm_tst.loc[14,'accuracy'],
                 acc_batch_norm_l2_tst.loc[14,'accuracy']]

methods = ['No Regularization,', 'L2', 'Dropout', 'Dropout + L2',
           'Dropout +  BatchNorm', 'Dropout +  BatchNorm + L2',
           'BatchNorm', 'BatchNorm + L2']

acc_table = pd.DataFrame(zip(methods,train_acc_list,test_acc_list),columns=["Method","Final Train Accuracy","Final Test Accuracy"])
print(acc_table)

###############################################################

print('Test Accuracy Comparrison: \n'
      'No Regularization : {0} , \n'
      'L2 : {1} ,\n'
      'Dropout : {2}, \n'
      'Dropout + L2 : {3}, \n'
      'Dropout +  BatchNorm : {4}, \n'
      'Dropout +  BatchNorm + L2 : {5}, \n'
      'BatchNorm : {6}, \n'
      'BatchNorm + L2 : {7}'.format(acc_none_tst.loc[14,'accuracy'],acc_l2_tst.loc[14,'accuracy'],
                                    acc_dropout_tst.loc[14,'accuracy'],acc_dropout_l2_tst.loc[14,'accuracy'],
                                    acc_dropout_batch_norm_tst.loc[14,'accuracy'],acc_dropout_batch_norm_l2_tst.loc[14,'accuracy'],
                                    acc_batch_norm_tst.loc[14,'accuracy'],acc_batch_norm_l2_tst.loc[14,'accuracy']))



# import pandas as pd
# import matplotlib.pyplot as plt
# plt.plot('epoch', 'accuracy', data=pd.DataFrame({'epoch':[1,2,3],'accuracy':[4,5,6]}), color='blue', markersize=12, linewidth=4, label= 'None')
# plt.plot('epoch', 'accuracy', data=pd.DataFrame({'epoch':[1,2,3],'accuracy':[4,5,6]}), color='red', markersize=12,linewidth=4, linestyle='dashed', label = 'L2 Test')
# plt.xlabel('epochs')
# plt.ylabel('accuracy')
# plt.title('Training Accuracy Per Epoch')
# plt.legend()
#
# plt.plot('epoch', 'accuracy', data=pd.DataFrame({'epoch':[1,2,3],'accuracy':[4,5,6]}), color='blue', markersize=12, linewidth=4, label= 'None')
# plt.plot('epoch', 'accuracy', data=pd.DataFrame({'epoch':[1,2,3],'accuracy':[4,5,6]}), color='red', markersize=12,linewidth=4, linestyle='dashed', label = 'L2 Test')
# plt.xlabel('epochs')
# plt.ylabel('accuracy')
# plt.title('Training Accuracy Per Epoch')
# plt.legend()
