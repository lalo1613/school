import torch
import numpy as np

def TestingNet(test_images, test_labels, NetName, dir_input):
    net = NetName()
    net.load_state_dict(torch.load(dir_input+"outputs/"+NetName().__class__.__name__+".pth"))
    test_images = test_images.reshape((test_images.shape[0], 1, test_images.shape[1], test_images.shape[1]))
    outputs = net(test_images)
    values, indices = torch.max(outputs, 1)
    preds = np.array(indices)
    test_labels = np.array(test_labels)
    accuracy = np.mean(preds == test_labels)
