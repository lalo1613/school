import  torch
from torch import nn
from torch.nn import functional as F
import gzip
import os
import re
import shutil
import pandas as pd
from mnist import MNIST
import numpy as np
import torch
from DLcourse.lenet5 import Net_BatchNorm, Net_Dropout, Net_Dropout_BatchNorm, Net_None
from DLcourse.testing import TestingNet
import matplotlib.pyplot as plt




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
    """
    :param encoder:
    :param decoder:
    :param x: input
    :param x_hat: target
    :param dim_img:
    :param dim_z:
    :param n_hidden:
    :param keep_prob:
    :return:
    """
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




###############################################################################
from mnist import MNIST

import torch
import numpy as np
import mnist_data
import os
import plot_utils
import glob

import argparse


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


##############
#plot_utils
import numpy as np
import matplotlib.pyplot as plt
from imageio import imwrite     # instead of imsave
from Pillow import imresize     # need to use resize instead


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
        images = images.reshape(self.n_img_x * self.n_img_y, self.img_h, self.img_w)
        imwrite(self.DIR + "/" + name, self._merge(images, [self.n_img_y, self.n_img_x]))

    def _merge(self, images, size):
        h, w = images.shape[1], images.shape[2]

        h_ = int(h * self.resize_factor)
        w_ = int(w * self.resize_factor)

        img = np.zeros((h_ * size[0], w_ * size[1]))

        for idx, image in enumerate(images):
            i = int(idx % size[1])
            j = int(idx / size[1])

            image_ = np.array(image.fromarray(w_,h_).resize())
            #image_ = imresize(image, size=(w_, h_), interp='bicubic')

            img[j * h_:j * h_ + h_, i * w_:i * w_ + w_] = image_

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

    def _merge(self, images, size):
        h, w = images.shape[1], images.shape[2]

        h_ = int(h * self.resize_factor)
        w_ = int(w * self.resize_factor)

        img = np.zeros((h_ * size[0], w_ * size[1]))

        for idx, image in enumerate(images):
            i = int(idx % size[1])
            j = int(idx / size[1])

            image_ = imresize(image, size=(w_, h_), interp='bicubic')

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
##############


IMAGE_SIZE_MNIST = 28




"""parsing and configuration"""


def parse_args():
    desc = "Pytorch implementation of 'Variational AutoEncoder (VAE)'"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--results_path', type=str, default='results',
                        help='File path of output images')

    parser.add_argument('--add_noise', type=bool, default=False,
                        help='Boolean for adding salt & pepper noise to input image')

    parser.add_argument('--dim_z', type=int, default='20', help='Dimension of latent vector', required=True)

    parser.add_argument('--n_hidden', type=int, default=500, help='Number of hidden units in MLP')

    parser.add_argument('--learn_rate', type=float, default=1e-3, help='Learning rate for Adam optimizer')

    parser.add_argument('--num_epochs', type=int, default=20, help='The number of epochs to run')

    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')

    parser.add_argument('--PRR', type=bool, default=True,
                        help='Boolean for plot-reproduce-result')

    parser.add_argument('--PRR_n_img_x', type=int, default=10,
                        help='Number of images along x-axis')

    parser.add_argument('--PRR_n_img_y', type=int, default=10,
                        help='Number of images along y-axis')

    parser.add_argument('--PRR_resize_factor', type=float, default=1.0,
                        help='Resize factor for each displayed image')

    parser.add_argument('--PMLR', type=bool, default=False,
                        help='Boolean for plot-manifold-learning-result')

    parser.add_argument('--PMLR_n_img_x', type=int, default=20,
                        help='Number of images along x-axis')

    parser.add_argument('--PMLR_n_img_y', type=int, default=20,
                        help='Number of images along y-axis')

    parser.add_argument('--PMLR_resize_factor', type=float, default=1.0,
                        help='Resize factor for each displayed image')

    parser.add_argument('--PMLR_z_range', type=float, default=2.0,
                        help='Range for unifomly distributed latent vector')

    parser.add_argument('--PMLR_n_samples', type=int, default=5000,
                        help='Number of samples in order to get distribution of labeled data')

    return check_args(parser.parse_args())



def check_args(args):
    # --results_path
    try:
        os.mkdir(args.results_path)
    except(FileExistsError):
        pass
    # delete all existing files
    files = glob.glob(args.results_path + '/*')
    for f in files:
        os.remove(f)

    # --add_noise
    try:
        assert args.add_noise == True or args.add_noise == False
    except:
        print('add_noise must be boolean type')
        return None

    # --dim-z
    try:
        assert args.dim_z > 0
    except:
        print('dim_z must be positive integer')
        return None

    # --n_hidden
    try:
        assert args.n_hidden >= 1
    except:
        print('number of hidden units must be larger than one')

    # --learn_rate
    try:
        assert args.learn_rate > 0
    except:
        print('learning rate must be positive')

    # --num_epochs
    try:
        assert args.num_epochs >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    # --PRR
    try:
        assert args.PRR == True or args.PRR == False
    except:
        print('PRR must be boolean type')
        return None

    if args.PRR == True:
        # --PRR_n_img_x, --PRR_n_img_y
        try:
            assert args.PRR_n_img_x >= 1 and args.PRR_n_img_y >= 1
        except:
            print('PRR : number of images along each axis must be larger than or equal to one')

        # --PRR_resize_factor
        try:
            assert args.PRR_resize_factor > 0
        except:
            print('PRR : resize factor for each displayed image must be positive')

    # --PMLR
    try:
        assert args.PMLR == True or args.PMLR == False
    except:
        print('PMLR must be boolean type')
        return None

    if args.PMLR == True:
        try:
            assert args.dim_z == 2
        except:
            print('PMLR : dim_z must be two')

        # --PMLR_n_img_x, --PMLR_n_img_y
        try:
            assert args.PMLR_n_img_x >= 1 and args.PMLR_n_img_y >= 1
        except:
            print('PMLR : number of images along each axis must be larger than or equal to one')

        # --PMLR_resize_factor
        try:
            assert args.PMLR_resize_factor > 0
        except:
            print('PMLR : resize factor for each displayed image must be positive')

        # --PMLR_z_range
        try:
            assert args.PMLR_z_range > 0
        except:
            print('PMLR : range for unifomly distributed latent vector must be positive')

        # --PMLR_n_samples
        try:
            assert args.PMLR_n_samples > 100
        except:
            print('PMLR : Number of samples in order to get distribution of labeled data must be large enough')

    return args





def main(args):


    # torch.manual_seed(222)
    # torch.cuda.manual_seed_all(222)
    # np.random.seed(222)


    device = torch.device('cuda')

    RESULTS_DIR = args.results_path
    ADD_NOISE = args.add_noise
    n_hidden = args.n_hidden
    dim_img = IMAGE_SIZE_MNIST ** 2  # number of pixels for a MNIST image
    dim_z = args.dim_z

    # train
    n_epochs = args.num_epochs
    batch_size = args.batch_size
    learn_rate = args.learn_rate

    # Plot
    PRR = args.PRR  # Plot Reproduce Result
    PRR_n_img_x = args.PRR_n_img_x  # number of images along x-axis in a canvas
    PRR_n_img_y = args.PRR_n_img_y  # number of images along y-axis in a canvas
    PRR_resize_factor = args.PRR_resize_factor  # resize factor for each image in a canvas

    PMLR = args.PMLR  # Plot Manifold Learning Result
    PMLR_n_img_x = args.PMLR_n_img_x  # number of images along x-axis in a canvas
    PMLR_n_img_y = args.PMLR_n_img_y  # number of images along y-axis in a canvas
    PMLR_resize_factor = args.PMLR_resize_factor  # resize factor for each image in a canvas
    PMLR_z_range = args.PMLR_z_range  # range for random latent vector
    PMLR_n_samples = args.PMLR_n_samples  # number of labeled samples to plot a map from input data space to the latent space

    """ prepare MNIST data """
    train_total_data, train_size, _, _, test_data, test_labels = mnist_data.prepare_MNIST_data()
    n_samples = train_size

    """ create network """
    keep_prob = 0.99
    encoder = vae.Encoder(dim_img, n_hidden, dim_z, keep_prob).to(device)
    decoder = vae.Decoder(dim_z, n_hidden, dim_img, keep_prob).to(device)
    # + operator will return but .extend is inplace no return.
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learn_rate)
    # vae.init_weights(encoder, decoder)

    """ training """
    # Plot for reproduce performance
    if PRR:
        PRR = plot_utils.Plot_Reproduce_Performance(RESULTS_DIR, PRR_n_img_x, PRR_n_img_y, IMAGE_SIZE_MNIST,
                                                    IMAGE_SIZE_MNIST, PRR_resize_factor)

        x_PRR = test_data[0:PRR.n_tot_imgs, :]

        x_PRR_img = x_PRR.reshape(PRR.n_tot_imgs, IMAGE_SIZE_MNIST, IMAGE_SIZE_MNIST)
        PRR.save_images(x_PRR_img, name='input.jpg')
        print('saved:', 'input.jpg')

        if ADD_NOISE:
            x_PRR = x_PRR * np.random.randint(2, size=x_PRR.shape)
            x_PRR += np.random.randint(2, size=x_PRR.shape)

            x_PRR_img = x_PRR.reshape(PRR.n_tot_imgs, IMAGE_SIZE_MNIST, IMAGE_SIZE_MNIST)
            PRR.save_images(x_PRR_img, name='input_noise.jpg')
            print('saved:', 'input_noise.jpg')

        x_PRR = torch.from_numpy(x_PRR).float().to(device)

    # Plot for manifold learning result
    if PMLR and dim_z == 2:

        PMLR = plot_utils.Plot_Manifold_Learning_Result(RESULTS_DIR, PMLR_n_img_x, PMLR_n_img_y, IMAGE_SIZE_MNIST,
                                                        IMAGE_SIZE_MNIST, PMLR_resize_factor, PMLR_z_range)

        x_PMLR = test_data[0:PMLR_n_samples, :]
        id_PMLR = test_labels[0:PMLR_n_samples, :]

        if ADD_NOISE:
            x_PMLR = x_PMLR * np.random.randint(2, size=x_PMLR.shape)
            x_PMLR += np.random.randint(2, size=x_PMLR.shape)


        z_ = torch.from_numpy(PMLR.z).float().to(device)
        x_PMLR = torch.from_numpy(x_PMLR).float().to(device)


    # train
    total_batch = int(n_samples / batch_size)
    min_tot_loss = np.inf
    for epoch in range(n_epochs):

        # Random shuffling
        np.random.shuffle(train_total_data)
        train_data_ = train_total_data[:, :-mnist_data.NUM_LABELS]

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

            batch_xs_input, batch_xs_target = torch.from_numpy(batch_xs_input).float().to(device), \
                                              torch.from_numpy(batch_xs_target).float().to(device)

            assert not torch.isnan(batch_xs_input).any()
            assert not torch.isnan(batch_xs_target).any()

            y, z, tot_loss, loss_likelihood, loss_divergence = \
                                        vae.get_loss(encoder, decoder, batch_xs_input, batch_xs_target)

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
                y_PRR = vae.get_ae(encoder, decoder, x_PRR)

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
                z_PMLR = vae.get_z(encoder, x_PMLR)
                PMLR.save_scattered_image(z_PMLR.detach().cpu().numpy(), id_PMLR,
                                          name="/PMLR_map_epoch_%02d" % (epoch) + ".jpg")
                print('saved:', "/PMLR_map_epoch_%02d" % (epoch) + ".jpg")


if __name__ == '__main__':

    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    # main
    main(args)