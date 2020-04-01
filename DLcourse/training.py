import torch
import numpy as np
import pandas as pd


def Training_LENET(train_images, train_labels, dir_input ,NetName, optimizer_input = None, n_epochs = 1):
    Net_dic = {'Net_None',  'Net_Dropout','Net_BatchNorm','Net_Dropout_BatchNorm'}
    net = NetName()
    # loading CNN (net's class defined in separate script)
    # defining a Loss function and parameter optimizer
    if optimizer_input is  None:
        optimizer_input = ''
        optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9 , weight_decay=0.01)

    criterion = torch.nn.CrossEntropyLoss()
    acc_epoc = []
    # training net
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        batch_size = 100
        for i in range(train_images.shape[0]//batch_size):
            # get the inputs; data is a list of [inputs, labels]
            inputs = train_images[(batch_size*i):(batch_size*(i+1))].apply_(float)
            labels = train_labels[(batch_size*i):(batch_size*(i+1))]
            inputs = inputs.reshape((batch_size,1,train_images.shape[1],train_images.shape[1]))

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            #print("loss "+str(running_loss/(i+1)))

         #print('outputs',torch.argmax(input = outputs, dim = 1), 'labels', labels)
        acc_epoc.append([epoch,np.mean(np.array(torch.argmax(input=outputs, dim=1)) == np.array(labels))])
        #acc_epoc = acc_epoc.append([epoch, np.mean(np.array(torch.argmax(input=outputs, dim=1)) == np.array(labels))])

    print('Finished Training')
    print('acc', acc_epoc)
    torch.save(net.state_dict(), dir_input+"outputs/"+NetName().__class__.__name__+optimizer_input+".pth")
    return pd.DataFrame(data = acc_epoc, columns = ['epoch','accuracy'])
