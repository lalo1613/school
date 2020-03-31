import gzip
import os
import re
import shutil
from mnist import MNIST
import numpy as np
import torch
from lenet5 import Net_BatchNorm
from lenet5 import Net_Dropout
from lenet5 import Net_Dropout_BatchNorm
from lenet5 import Net_None



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

Training_LENET(train_images = train_images, train_labels = train_labels, dir_input = dir_input ,NetName = Net_Dropout,optimizer= None)


#
# # loading CNN (net's class defined in separate script)
# net = Net_None()
#
# # defining a Loss function and parameter optimizer
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# optimizer_l2 = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9 , weight_decay=0.01)
#
#
# # training net
# for epoch in range(10):  # loop over the dataset multiple times
#     running_loss = 0.0
#     batch_size = 1000
#     for i in range(train_images.shape[0]//batch_size):
#         # get the inputs; data is a list of [inputs, labels]
#         inputs = train_images[(batch_size*i):(batch_size*(i+1))].apply_(float)
#         labels = train_labels[(batch_size*i):(batch_size*(i+1))]
#         inputs = inputs.reshape((batch_size,1,train_images.shape[1],train_images.shape[1]))
#
#         # zero the parameter gradients
#         optimizer.zero_grad()
#
#         # forward + backward + optimize
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         # print statistics
#         running_loss += loss.item()
#         print("loss "+str(running_loss/(i+1)))
#
# print('Finished Training')
#
# torch.save(net.state_dict(), dir_input+"outputs/model.pth")

# Testing

net = Net_None()
net.load_state_dict(torch.load(dir_input+"outputs/model.pth"))
test_images = test_images.reshape((test_images.shape[0],1,test_images.shape[1],test_images.shape[1]))
outputs = net(test_images)
values, indices = torch.max(outputs, 1)
preds = np.array(indices)
test_labels = np.array(test_labels)
acc_none = np.mean(preds == test_labels)

net = Net_BatchNorm()
net.load_state_dict(torch.load(dir_input+"outputs/model.pth"))
test_images = test_images.reshape((test_images.shape[0],1,test_images.shape[1],test_images.shape[1]))
outputs = net(test_images)
values, indices = torch.max(outputs, 1)
preds = np.array(indices)
test_labels = np.array(test_labels)
acc_BatchNorm = np.mean(preds == test_labels)

net = Net_Dropout()
net.load_state_dict(torch.load(dir_input+"outputs/model.pth"))
test_images = test_images.reshape((test_images.shape[0],1,test_images.shape[1],test_images.shape[1]))
outputs = net(test_images)
values, indices = torch.max(outputs, 1)
preds = np.array(indices)
test_labels = np.array(test_labels)
acc_Dropout = np.mean(preds == test_labels)

net = Net_Dropout_BatchNorm()
net.load_state_dict(torch.load(dir_input+"outputs/model.pth"))
test_images = test_images.reshape((test_images.shape[0],1,test_images.shape[1],test_images.shape[1]))
outputs = net(test_images)
values, indices = torch.max(outputs, 1)
preds = np.array(indices)
test_labels = np.array(test_labels)
acc_Dropout_BatchNorm = np.mean(preds == test_labels)
