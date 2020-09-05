import gzip
import os
import re
import shutil
import pandas as pd
from mnist import MNIST
import numpy as np
import torch
from DLcourse.lenet5 import Net_BatchNorm, Net_Dropout, Net_Dropout_BatchNorm, Net_None
#from DLcourse.training import Training_LENET
from DLcourse.testing import TestingNet
import matplotlib.pyplot as plt



# setting directories now
dir_input = r"C:\Users\Bengal\Downloads\FashionMNIST"+"\\"
dir_uncompressed = r"C:\Users\Bengal\Downloads\FashionMNIST\uncompressed"+"\\"

# uncompressing image files and moving them to separate directory (run once only!)
# file_list = os.listdir(dir_input)
# for f in file_list:
#     with gzip.open(dir_input+f) as gz:
#         with open(dir_uncompressed+re.sub('.gz','',f),'wb') as to_save:
#             shutil.copyfileobj(gz,to_save)

# loading train and test data
mndata = MNIST(dir_uncompressed)
train_images, train_labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()

# reshaping images into 2d arrays
train_images = np.array(train_images).reshape((60000,28,28))
test_images = np.array(test_images).reshape((10000,28,28))

# saving train and test data as torch tensors
train_images = torch.tensor(train_images).float()
train_labels = torch.tensor(train_labels)
test_images = torch.tensor(test_images).float()
test_labels = torch.tensor(test_labels)

#  None
acc_none, acc_none_tst = Training_LENET(train_images = train_images, train_labels = train_labels,test_images = test_images, test_labels = test_labels, dir_input = dir_input ,NetName = Net_None,optimizer_input= None, n_epochs = 15)
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
