# Some code was borrowed from https://github.com/petewarden/tensorflow_makefile/blob/master/tensorflow/models/image/mnist/convolutional.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
NUM_LABELS = 10
VALIDATION_SIZE = 5000  # Size of the validation set.

dir_input = r"C:\Users\Bengal\Downloads\FashionMNIST" +"\\"
dir_uncompressed = r"C:\Users\Bengal\Downloads\FashionMNIST\uncompressed" #+"\\"
#DATA_DIRECTORY = dir_uncompressed
DATA_DIRECTORY = r"C:\Users\Bengal\Downloads\FashionMNIST"
# tf.io.gfile.exists(DATA_DIRECTORY)
# tf.io.gfile.exists(os.path.join(DATA_DIRECTORY, 'train-images-idx3-ubyte'))

# Download MNIST data
# def maybe_download(filename):
#     """Download the data from Yann's website, unless it's already here."""
#     #if not tf.io.gfile.exists(DATA_DIRECTORY):
#     #    tf.io.gfile.MakeDirs(DATA_DIRECTORY)
#     filepath = os.path.join(DATA_DIRECTORY, filename)
#     if not tf.io.gfile.exists(filepath):
#         filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
#         with tf.io.gfile.GFile(filepath) as f:
#             size = f.size()
#         print('Successfully downloaded', filename, size, 'bytes.')
#     return filepath


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
        num_labels_data = len(labels)
        one_hot_encoding = numpy.zeros((num_labels_data, NUM_LABELS))
        one_hot_encoding[numpy.arange(num_labels_data), labels] = 1
        one_hot_encoding = numpy.reshape(one_hot_encoding, [-1, NUM_LABELS])
    return one_hot_encoding


# Augment training data
def expend_training_data(images, labels):
    expanded_images = []
    expanded_labels = []

    j = 0  # counter
    for x, y in zip(images, labels):
        j = j + 1
        if j % 100 == 0:
            print('expanding data : %03d / %03d' % (j, numpy.size(images, 0)))

        # register original data
        expanded_images.append(x)
        expanded_labels.append(y)

        # get a value for the background
        # zero is the expected value, but median() is used to estimate background's value
        bg_value = numpy.median(x)  # this is regarded as background's value
        image = numpy.reshape(x, (-1, 28))

        for i in range(4):
            # rotate the image with random degree
            angle = numpy.random.randint(-15, 15, 1)
            new_img = ndimage.rotate(image, angle, reshape=False, cval=bg_value)

            # shift the image with random distance
            shift = numpy.random.randint(-2, 2, 2)
            new_img_ = ndimage.shift(new_img, shift, cval=bg_value)

            # register new training data
            expanded_images.append(numpy.reshape(new_img_, 784))
            expanded_labels.append(y)

    # images and labels are concatenated for random-shuffle at each epoch
    # notice that pair of image and label should not be broken
    expanded_train_total_data = numpy.concatenate((expanded_images, expanded_labels), axis=1)
    numpy.random.shuffle(expanded_train_total_data)

    return expanded_train_total_data


# Prepare MNISt data
def prepare_MNIST_data(use_norm_shift=False, use_norm_scale=True, use_data_augmentation=False):
    # Get the data.
    train_data_filename = os.path.join(DATA_DIRECTORY, 'train-images-idx3-ubyte.gz')
    train_labels_filename = os.path.join(DATA_DIRECTORY,'train-labels-idx1-ubyte.gz')
    test_data_filename = os.path.join(DATA_DIRECTORY,'t10k-images-idx3-ubyte.gz')
    test_labels_filename = os.path.join(DATA_DIRECTORY,'t10k-labels-idx1-ubyte.gz')
    # train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
    # test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
    # test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')

    # Extract it into numpy arrays.
    train_data = extract_data(train_data_filename, 60000, use_norm_shift, use_norm_scale)
    train_labels = extract_labels(train_labels_filename, 60000)
    test_data = extract_data(test_data_filename, 10000, use_norm_shift, use_norm_scale)
    test_labels = extract_labels(test_labels_filename, 10000)

    # Generate a validation set.
    validation_data = train_data[:VALIDATION_SIZE, :]
    validation_labels = train_labels[:VALIDATION_SIZE, :]
    train_data = train_data[VALIDATION_SIZE:, :]
    train_labels = train_labels[VALIDATION_SIZE:, :]

    # Concatenate train_data & train_labels for random shuffle
    if use_data_augmentation:
        train_total_data = expend_training_data(train_data, train_labels)
    else:
        train_total_data = numpy.concatenate((train_data, train_labels), axis=1)

    train_size = train_total_data.shape[0]

    return train_total_data, train_size, validation_data, validation_labels, test_data, test_labels


################ VAE

import  torch
from    torch import nn
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


class Decoder(torch.nn.Module):
    def __init__(self, D_in, H, D_out, keep_prob=0):
        super(Decoder, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return F.relu(self.linear2(x))

def get_ae(encoder, decoder, x):
    # encoding
    mu, log_sigma = encoder(x)
    sigma = torch.exp(log_sigma)
    # sampling by re-parameterization technique
    z = mu + sigma * torch.randn_like(mu)

    # decoding
    y = decoder(z)
    # y = torch.clamp(y, 1e-8, 1 - 1e-8)

    return y

def get_z(encoder, x):

    # encoding
    mu, log_sigma = encoder(x)
    sigma = torch.exp(log_sigma)
    # sampling by re-parameterization technique
    z = mu + sigma * torch.randn_like(mu)

    return z


def get_loss(encoder, decoder, x, x_target):
    batchsz = x.size(0)
    # encoding
    mu, log_sigma = encoder(x)
    sigma = torch.exp(log_sigma)
    # sampling by re-parameterization technique
    z = mu + sigma * torch.randn_like(mu)

    # decoding
    y = decoder(z)
    # y = torch.clamp(y, 1e-8, 1 - 1e-8)


    # loss
    # marginal_likelihood2 = torch.sum(x_target * torch.log(y) + (1 - x_target) * torch.log(1 - y)) / batchsz
    # marginal_likelihood = -F.binary_cross_entropy(y, x_target, reduction='sum') / batchsz
    marginal_likelihood = -torch.pow(x_target - y, 2).sum() / batchsz
    # print(marginal_likelihood2.item(), marginal_likelihood.item())

    KL_divergence = 0.5 * torch.sum(
                                torch.pow(mu, 2) +
                                torch.pow(sigma, 2) -
                                torch.log(1e-8 + torch.pow(sigma, 2)) - 1
                               ).sum() / batchsz

    ELBO = marginal_likelihood - KL_divergence

    loss = -ELBO

    return y, z, loss, marginal_likelihood, KL_divergence

#### PLOT

import numpy as np
import matplotlib.pyplot as plt
#from scipy.misc import imsave
from imageio import imwrite     # instead of imsave
#from scipy.misc import imresize


class Plot_Reproduce_Performance():
    def __init__(self, DIR, n_img_x=8, n_img_y=8, img_w=28, img_h=28, resize_factor=1.0):
        self.DIR = DIR

        assert n_img_x > 0 and n_img_y > 0

        self.n_img_x = n_img_x
        self.n_img_y = n_img_y
        self.n_tot_imgs = n_img_x * n_img_y

        assert img_w > 0 and img_h > 0

        self.img_w = img_w
        self.img_h = img_h

        assert resize_factor > 0

        self.resize_factor = resize_factor

    def save_images(self, images, name='result.jpg'):
        # plt.imshow(images[3])
        images = images.reshape(self.n_img_x * self.n_img_y, self.img_h, self.img_w)
        #imsave(self.DIR + "/" + name, self._merge(images, [self.n_img_y, self.n_img_x]))
        #plt.imshow(images)
        imwrite(self.DIR + "/" + name, self._merge(images, [self.n_img_y, self.n_img_x]))

    def _merge(self, images, size):
        h, w = images.shape[1], images.shape[2]

        h_ = int(h * self.resize_factor)
        w_ = int(w * self.resize_factor)

        img = np.zeros((h_ * size[0], w_ * size[1]))

        for idx, image in enumerate(images):
            i = int(idx % size[1])
            j = int(idx / size[1])
            #image_ = image.resize(w_, h_)
            #print(images.shape[1], images.shape[2])
            #image_ = image.resize(images.shape[1], images.shape[2])

            image_ = image
            #image_ = np.array(image.resize(w_, h_))
            #image_ = imresize(image, size=(w_, h_), interp='bicubic')

            img[j * h_:j * h_ + h_, i * w_:i * w_ + w_] = image_
            #plt.imshow(img)
        return img


class Plot_Manifold_Learning_Result():
    def __init__(self, DIR, n_img_x=20, n_img_y=20, img_w=28, img_h=28, resize_factor=1.0, z_range=4):
        self.DIR = DIR

        assert n_img_x > 0 and n_img_y > 0

        self.n_img_x = n_img_x
        self.n_img_y = n_img_y
        self.n_tot_imgs = n_img_x * n_img_y

        assert img_w > 0 and img_h > 0

        self.img_w = img_w
        self.img_h = img_h

        assert resize_factor > 0

        self.resize_factor = resize_factor

        assert z_range > 0
        self.z_range = z_range

        self._set_latent_vectors()

    def _set_latent_vectors(self):
        # z1 = np.linspace(-self.z_range, self.z_range, self.n_img_y)
        # z2 = np.linspace(-self.z_range, self.z_range, self.n_img_x)
        #
        # z = np.array(np.meshgrid(z1, z2))
        # z = z.reshape([-1, 2])

        # borrowed from https://github.com/fastforwardlabs/vae-tf/blob/master/plot.py
        z = np.rollaxis(
            np.mgrid[self.z_range:-self.z_range:self.n_img_y * 1j, self.z_range:-self.z_range:self.n_img_x * 1j], 0, 3)
        # z1 = np.rollaxis(np.mgrid[1:-1:self.n_img_y * 1j, 1:-1:self.n_img_x * 1j], 0, 3)
        # z = z1**2
        # z[z1<0] *= -1
        #
        # z = z*self.z_range

        self.z = z.reshape([-1, 2])

    def save_images(self, images, name='result.jpg'):
        images = images.reshape(self.n_img_x * self.n_img_y, self.img_h, self.img_w)
        imwrite(self.DIR + "/" + name, self._merge(images, [self.n_img_y, self.n_img_x]))
        #imsave(self.DIR + "/" + name, self._merge(images, [self.n_img_y, self.n_img_x]))

    def _merge(self, images, size):
        h, w = images.shape[1], images.shape[2]

        h_ = int(h * self.resize_factor)
        w_ = int(w * self.resize_factor)

        img = np.zeros((h_ * size[0], w_ * size[1]))

        for idx, image in enumerate(images):
            i = int(idx % size[1])
            j = int(idx / size[1])

            image_ = np.array(image.resize(w_, h_))
            #image_ = imresize(image, size=(w_, h_), interp='bicubic')

            img[j * h_:j * h_ + h_, i * w_:i * w_ + w_] = image_

        return img

    # borrowed from https://github.com/ykwon0407/variational_autoencoder/blob/master/variational_bayes.ipynb
    def save_scattered_image(self, z, id, name='scattered_image.jpg'):
        N = 10
        plt.figure(figsize=(8, 6))
        plt.scatter(z[:, 0], z[:, 1], c=np.argmax(id, 1), marker='o', edgecolor='none', cmap=discrete_cmap(N, 'jet'))
        plt.colorbar(ticks=range(N))
        axes = plt.gca()
        axes.set_xlim([-self.z_range - 2, self.z_range + 2])
        axes.set_ylim([-self.z_range - 2, self.z_range + 2])
        plt.grid(True)
        plt.savefig(self.DIR + "/" + name)


# borrowed from https://gist.github.com/jakevdp/91077b0cae40f8f8244a
def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

#### MAIN

import torch
import numpy as np
import os
import glob


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

def main(results_path,add_noise,dim_z,n_hidden,learn_rate,num_epochs,batch_size,PRR,PRR_n_img_x,PRR_n_img_y,PRR_resize_factor,
         PMLR,PMLR_n_img_x,PMLR_n_img_y,PMLR_resize_factor,PMLR_z_range,PMLR_n_samples):


    # torch.manual_seed(222)
    # torch.cuda.manual_seed_all(222)
    # np.random.seed(222)


    #device = torch.device('cuda')

    RESULTS_DIR = results_path
    ADD_NOISE = add_noise
    n_hidden = n_hidden
    dim_img = IMAGE_SIZE_MNIST ** 2  # number of pixels for a MNIST image
    #dim_z = dim_z

    # train
    n_epochs = num_epochs
    #batch_size = batch_size
    #learn_rate = learn_rate

    # Plot
    # PRR = PRR  # Plot Reproduce Result
    # PRR_n_img_x = PRR_n_img_x  # number of images along x-axis in a canvas
    # PRR_n_img_y = PRR_n_img_y  # number of images along y-axis in a canvas
    # PRR_resize_factor = PRR_resize_factor  # resize factor for each image in a canvas
    #
    # PMLR = PMLR  # Plot Manifold Learning Result
    # PMLR_n_img_x = PMLR_n_img_x  # number of images along x-axis in a canvas
    # PMLR_n_img_y = PMLR_n_img_y  # number of images along y-axis in a canvas
    # PMLR_resize_factor = PMLR_resize_factor  # resize factor for each image in a canvas
    # PMLR_z_range = PMLR_z_range  # range for random latent vector
    # PMLR_n_samples = PMLR_n_samples  # number of labeled samples to plot a map from input data space to the latent space

    """ prepare MNIST data """
    #train_total_data, train_size, _, _, test_data, test_labels = mnist_data.prepare_MNIST_data()
    train_total_data, train_size, _, _, test_data, test_labels = prepare_MNIST_data()
    n_samples = train_size

    """ create network """
    keep_prob = 0.99
    encoder = Encoder(dim_img, n_hidden, dim_z, keep_prob)#.to(device)
    decoder = Decoder(dim_z, n_hidden, dim_img, keep_prob)#.to(device)
    # + operator will return but .extend is inplace no return.
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learn_rate)
    # vae.init_weights(encoder, decoder)

    """ training """
    # Plot for reproduce performance
    if PRR:
        PRR = Plot_Reproduce_Performance(RESULTS_DIR, PRR_n_img_x, PRR_n_img_y, IMAGE_SIZE_MNIST,
                                                    IMAGE_SIZE_MNIST, PRR_resize_factor)

        x_PRR = test_data[0:PRR.n_tot_imgs, :]

        x_PRR_img = x_PRR.reshape(PRR.n_tot_imgs, IMAGE_SIZE_MNIST, IMAGE_SIZE_MNIST)
        #print(x_PRR_img)
        PRR.save_images(x_PRR_img, name='input.jpg')
        print('saved:', 'input.jpg')

        if ADD_NOISE:
            x_PRR = x_PRR * np.random.randint(2, size=x_PRR.shape)
            x_PRR += np.random.randint(2, size=x_PRR.shape)

            x_PRR_img = x_PRR.reshape(PRR.n_tot_imgs, IMAGE_SIZE_MNIST, IMAGE_SIZE_MNIST)
            PRR.save_images(x_PRR_img, name='input_noise.jpg')
            print('saved:', 'input_noise.jpg')

        #x_PRR = x_PRR.float()#.to(device)
        x_PRR = torch.from_numpy(x_PRR).float()#.to(device)

    # Plot for manifold learning result
    if PMLR and dim_z == 2:

        PMLR = Plot_Manifold_Learning_Result(RESULTS_DIR, PMLR_n_img_x, PMLR_n_img_y, IMAGE_SIZE_MNIST,
                                                        IMAGE_SIZE_MNIST, PMLR_resize_factor, PMLR_z_range)

        x_PMLR = test_data[0:PMLR_n_samples, :]
        id_PMLR = test_labels[0:PMLR_n_samples, :]

        if ADD_NOISE:
            x_PMLR = x_PMLR * np.random.randint(2, size=x_PMLR.shape)
            x_PMLR += np.random.randint(2, size=x_PMLR.shape)


        z_ = torch.from_numpy(PMLR.z).float()#.to(device)
        x_PMLR = torch.from_numpy(x_PMLR).float()#.to(device)


    # train
    total_batch = int(n_samples / batch_size)
    min_tot_loss = np.inf
    for epoch in range(n_epochs):

        # Random shuffling
        np.random.shuffle(train_total_data)
        #train_data_ = train_total_data[:, :-mnist_data.NUM_LABELS]
        train_data_ = train_total_data[:, :-NUM_LABELS]

        # Loop over all batches
        encoder.train()
        decoder.train()
        for i in range(total_batch):
            # Compute the offset of the current minibatch in the data.
            offset = (i * batch_size) % (n_samples)
            batch_xs_input = train_data_[offset:(offset + batch_size), :]

            batch_xs_target = batch_xs_input

            # add salt & pepper noise
            if ADD_NOISE:
                batch_xs_input = batch_xs_input * np.random.randint(2, size=batch_xs_input.shape)
                batch_xs_input += np.random.randint(2, size=batch_xs_input.shape)

            #batch_xs_input, batch_xs_target = torch.from_numpy(batch_xs_input).float().to(device),torch.from_numpy(batch_xs_target).float().to(device)
            batch_xs_input, batch_xs_target = torch.from_numpy(batch_xs_input).float(),torch.from_numpy(batch_xs_target).float()

            assert not torch.isnan(batch_xs_input).any()
            assert not torch.isnan(batch_xs_target).any()

            y, z, tot_loss, loss_likelihood, loss_divergence = \
                                        get_loss(encoder, decoder, batch_xs_input, batch_xs_target)

            optimizer.zero_grad()
            tot_loss.backward()
            optimizer.step()



            # print cost every epoch
        print("epoch %d: L_tot %03.2f L_likelihood %03.2f L_divergence %03.2f" % (
                                                epoch, tot_loss.item(), loss_likelihood.item(), loss_divergence.item()))




        encoder.eval()
        decoder.eval()
        # if minimum loss is updated or final epoch, plot results
        if min_tot_loss > tot_loss.item() or epoch + 1 == n_epochs:
            min_tot_loss = tot_loss.item()

            # Plot for reproduce performance
            if PRR:
                y_PRR = get_ae(encoder, decoder, x_PRR)

                y_PRR_img = y_PRR.reshape(PRR.n_tot_imgs, IMAGE_SIZE_MNIST, IMAGE_SIZE_MNIST)
                PRR.save_images(y_PRR_img.detach().cpu().numpy(), name="/PRR_epoch_%02d" % (epoch) + ".jpg")
                print('saved:', "/PRR_epoch_%02d" % (epoch) + ".jpg")

            # Plot for manifold learning result
            if PMLR and dim_z == 2:
                y_PMLR = decoder(z_)

                y_PMLR_img = y_PMLR.reshape(PMLR.n_tot_imgs, IMAGE_SIZE_MNIST, IMAGE_SIZE_MNIST)
                PMLR.save_images(y_PMLR_img.detach().cpu().numpy(), name="/PMLR_epoch_%02d" % (epoch) + ".jpg")
                print('saved:', "/PMLR_epoch_%02d" % (epoch) + ".jpg")

                # plot distribution of labeled images
                z_PMLR = get_z(encoder, x_PMLR)
                PMLR.save_scattered_image(z_PMLR.detach().cpu().numpy(), id_PMLR,
                                          name="/PMLR_map_epoch_%02d" % (epoch) + ".jpg")
                print('saved:', "/PMLR_map_epoch_%02d" % (epoch) + ".jpg")



main(results_path = results_path,add_noise = add_noise,dim_z = dim_z,n_hidden = n_hidden,learn_rate = learn_rate,num_epochs = num_epochs,
     batch_size = batch_size,PRR = PRR,PRR_n_img_x = PRR_n_img_x,PRR_n_img_y = PRR_n_img_y,PRR_resize_factor = PRR_resize_factor,
     PMLR = PMLR,PMLR_n_img_x = PMLR_n_img_x,PMLR_n_img_y = PMLR_n_img_y,PMLR_resize_factor = PMLR_resize_factor,PMLR_z_range = PMLR_z_range,PMLR_n_samples = PMLR_n_samples)
