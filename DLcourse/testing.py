import torch
import numpy as np
from DLcourse.lenet5 import Net

def TestingNet(test_images, test_labels, dir_input, optimizer_input = None):
    if optimizer_input is  None:
        optimizer_input = ''
    else:
        optimizer_input = 'l2'
    net = Net
    net.load_state_dict(torch.load(dir_input+"outputs\\"+Net.__class__.__name__+str(optimizer_input)+".pth"))
    test_images = test_images.reshape((test_images.shape[0], 1, test_images.shape[1], test_images.shape[1]))
    outputs = net(test_images)
    values, indices = torch.max(outputs, 1)
    preds = np.array(indices)
    test_labels = np.array(test_labels)
    return np.mean(preds == test_labels)
