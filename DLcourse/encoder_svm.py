# Some code was borrowed from https://github.com/petewarden/tensorflow_makefile/blob/master/tensorflow/models/image/mnist/convolutional.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import gzip
import os

import numpy
from scipy import ndimage

from six.moves import urllib

import tensorflow as tf

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
#DATA_DIRECTORY = "data"

# Params for MNIST
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 1 ## changed from one hot encoding to labels
VALIDATION_SIZE = 5000  # Size of the validation set.

dir_input = r"C:\Users\Bengal\Downloads\FashionMNIST" +"\\"
dir_uncompressed = r"C:\Users\Bengal\Downloads\FashionMNIST\uncompressed" #+"\\"
DATA_DIRECTORY = r"C:\Users\Bengal\Downloads\FashionMNIST"

# Extract the images
def extract_data(filename, num_images, norm_shift=False, norm_scale=True):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
        data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
        if norm_shift:
            data = data - (PIXEL_DEPTH / 2.0)
        if norm_scale:
            data = data / PIXEL_DEPTH
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
        data = numpy.reshape(data, [num_images, -1])
    return data


# Extract the labels
def extract_labels(filename, num_images):
    """Extract the labels into a vector of int64 label IDs."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.int64)
        #print(labels)
        num_labels_data = len(labels)
        one_hot_encoding = numpy.zeros((num_labels_data, NUM_LABELS))
        one_hot_encoding[numpy.arange(num_labels_data), labels] = 1
        one_hot_encoding = numpy.reshape(one_hot_encoding, [-1, NUM_LABELS])
    #return one_hot_encoding
    return labels

# Prepare FASHION MNIST data
def prepare_F_MNIST_data(use_norm_shift=False, use_norm_scale=True, use_data_augmentation=False):
    # Get the data.
    train_data_filename = os.path.join(DATA_DIRECTORY, 'train-images-idx3-ubyte.gz')
    train_labels_filename = os.path.join(DATA_DIRECTORY,'train-labels-idx1-ubyte.gz')
    test_data_filename = os.path.join(DATA_DIRECTORY,'t10k-images-idx3-ubyte.gz')
    test_labels_filename = os.path.join(DATA_DIRECTORY,'t10k-labels-idx1-ubyte.gz')

    # Extract it into numpy arrays.
    train_data = extract_data(train_data_filename, 60000, use_norm_shift, use_norm_scale)
    train_labels = extract_labels(train_labels_filename, 60000)
    test_data = extract_data(test_data_filename, 10000, use_norm_shift, use_norm_scale)
    test_labels = extract_labels(test_labels_filename, 10000)

    # Generate a validation set.
    validation_data = train_data[:VALIDATION_SIZE, :]
    #validation_labels = train_labels[:VALIDATION_SIZE, :]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_data = train_data[VALIDATION_SIZE:, :]
    #train_labels = train_labels[VALIDATION_SIZE:, :]
    train_labels = train_labels[VALIDATION_SIZE:]

    #train_total_data = numpy.concatenate((train_data, train_labels), axis=1)

    #train_size = train_total_data.shape[0]
    train_size = train_data.shape[0]

    return  train_size, validation_data, validation_labels, test_data, test_labels, train_data, train_labels #, train_total_data

################ VAE

import  torch
from    torch.nn import functional as F


class Encoder(torch.nn.Module):
    def __init__(self, D_in, H, D_out, keep_prob=0):
        super(Encoder, self).__init__()

        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, 32)

        self._enc_mu = torch.nn.Linear(32, D_out)
        self._enc_log_sigma = torch.nn.Linear(32, D_out)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))

        # print(x.shape)

        return self._enc_mu(x), self._enc_log_sigma(x)


def SVM(X, y):
    X = X.detach().numpy()
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(X, y)
    return clf.predict_proba()



def get_z(encoder, x):

    # encoding
    mu, log_sigma = encoder(x)
    sigma = torch.exp(log_sigma)
    # sampling by re-parameterization technique
    z = mu + sigma * torch.randn_like(mu)

    return z


def get_loss(encoder, SVM, x, x_target):
    batchsz = x.size(0)
    # encoding
    mu, log_sigma = encoder(x)
    sigma = torch.exp(log_sigma)
    # sampling by re-parameterization technique
    z = mu + sigma * torch.randn_like(mu)

    # SVM
    y = SVM(z,x_target)


    # loss
    marginal_likelihood = -torch.pow(x_target - y, 2).sum() / batchsz

    KL_divergence = 0.5 * torch.sum(
                                torch.pow(mu, 2) +
                                torch.pow(sigma, 2) -
                                torch.log(1e-8 + torch.pow(sigma, 2)) - 1
                               ).sum() / batchsz

    ELBO = marginal_likelihood - KL_divergence

    loss = -ELBO

    return y, z, loss, marginal_likelihood, KL_divergence


#### MAIN

import torch
import numpy as np
import os


IMAGE_SIZE_MNIST = 28

IMAGE_SIZE_MNIST = 28
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
#VALIDATION_SIZE = 5000  # Size of the validation set.



results_path = dir_input + 'results'
add_noise = False
dim_z = 20
n_hidden = 500
learn_rate = 1e-3
num_epochs = 1
batch_size = 128
PRR = True
PRR_n_img_x = 10
PRR_n_img_y = 10
PRR_resize_factor = 1.0
PMLR = False
PMLR_n_img_x = 20
PMLR_n_img_y = 20
PMLR_resize_factor = 1.0
PMLR_z_range = 2.0
PMLR_n_samples = 5000

############################
# CHECK WHY THE SVM DONT RUN WITH THE INPUT I GIVE IT
#train_size, _, _, test_data, test_labels, train_data, train_labels, train_total_data = prepare_F_MNIST_data()
train_size, _, _, test_data, test_labels, train_data, train_labels = prepare_F_MNIST_data()

#X = torch.from_numpy(train_total_data[:100, :-NUM_LABELS]).float()
X = train_data
y = train_labels
#try to concatinate X and y for shuffling - dont succeed!!
X = list(X)
y = list(y)
np.concatenate((train_data, train_labels), axis=1)
#y = torch.from_numpy(train_total_data[:100, -NUM_LABELS:]).float()
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
#X = np.array(X)
#y = np.array(y)
tf.size(y)
clf.fit(X, y)
clf.predict_proba()
###########################



def main(results_path,add_noise,dim_z,n_hidden,learn_rate,num_epochs,batch_size,PRR,PRR_n_img_x,PRR_n_img_y,PRR_resize_factor,
         PMLR,PMLR_n_img_x,PMLR_n_img_y,PMLR_resize_factor,PMLR_z_range,PMLR_n_samples):

    dim_img = IMAGE_SIZE_MNIST ** 2  # number of pixels for a MNIST image

    """ prepare MNIST data """
    #train_size, _, _, test_data, test_labels, train_date, train_labels, train_total_data  = prepare_F_MNIST_data()
    train_size, _, _, test_data, test_labels, train_date, train_labels = prepare_F_MNIST_data()
    n_samples = train_size

    """ create network """
    keep_prob = 0.99
    encoder = Encoder(dim_img, n_hidden, dim_z, keep_prob)
    svm = SVC(kernel = 'linear', C = 1)
    # + operator will return but .extend is inplace no return.
    optimizer = torch.optim.Adam(list(encoder.parameters()), lr=learn_rate)

    # train
    total_batch = int(n_samples / batch_size)
    min_tot_loss = np.inf
    for epoch in range(num_epochs):

        # Random shuffling
        train_total_data = np.concatenate((train_data, train_labels), axis=1)
        np.random.shuffle(train_total_data)
        train_data_ = train_total_data[:, :-NUM_LABELS]

        # Loop over all batches
        encoder.train()
        for i in range(total_batch):
            # Compute the offset of the current minibatch in the data.
            offset = (i * batch_size) % (n_samples)
            batch_xs_input = train_data_[offset:(offset + batch_size), :]
            batch_xs_target = batch_xs_input
            batch_train_labels = torch.from_numpy(train_total_data[offset:(offset + batch_size), -NUM_LABELS:]).float()
            batch_xs_input, batch_xs_target = torch.from_numpy(batch_xs_input).float(),torch.from_numpy(batch_xs_target).float()

            assert not torch.isnan(batch_xs_input).any()
            assert not torch.isnan(batch_xs_target).any()

            y, z, tot_loss, loss_likelihood, loss_divergence = get_loss(encoder, SVM, batch_xs_input, batch_train_labels)



            optimizer.zero_grad()
            tot_loss.backward()
            optimizer.step()

            # print cost every epoch
        print("epoch %d: L_tot %03.2f L_likelihood %03.2f L_divergence %03.2f" % (
                                                epoch, tot_loss.item(), loss_likelihood.item(), loss_divergence.item()))

        encoder.eval()

main(results_path = results_path,add_noise = add_noise,dim_z = dim_z,n_hidden = n_hidden,learn_rate = learn_rate,num_epochs = num_epochs,
     batch_size = batch_size,PRR = PRR,PRR_n_img_x = PRR_n_img_x,PRR_n_img_y = PRR_n_img_y,PRR_resize_factor = PRR_resize_factor,
     PMLR = PMLR,PMLR_n_img_x = PMLR_n_img_x,PMLR_n_img_y = PMLR_n_img_y,PMLR_resize_factor = PMLR_resize_factor,PMLR_z_range = PMLR_z_range,PMLR_n_samples = PMLR_n_samples)