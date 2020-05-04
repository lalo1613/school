import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F



def Training_LSTM(train_input, train_output, test_input, test_output, dir_input ,NetName,dataset_lengths,vocabulary, n_epochs = 1):
    Net_dic = {'Net_LSTM',  'Net_LSTM_drop','Net_GRU','Net_GRU_drop'}
    net = NetName(vocabulary = vocabulary, dataset_lengths = dataset_lengths)

    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    #else:
    #    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9 , weight_decay=0.01)

    acc_epoc = []
    acc_epoc_test = []
    # training net
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        batch_size = 1024
        for i in range((len(train_input)//batch_size)+1):
            # get the inputs; data is a list of [inputs, labels]
            inputs = train_input[(batch_size*i):(batch_size*(i+1))]#.apply_(float)
            outputs = train_output[(batch_size*i):(batch_size*(i+1))]
            #inputs = inputs.reshape((batch_size,1,train_input.shape[1],train_input.shape[1]))

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            preds = net(inputs,vocabulary, dataset_lengths)
            # getting loss using cross entropy
            loss = F.cross_entropy(preds, outputs)

            # calculating perplexity
            perplexity = torch.exp(loss)
            print('Loss:', loss, 'PP:', perplexity)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            #print("loss "+str(running_loss/(i+1)))

         #print('outputs',torch.argmax(input = outputs, dim = 1), 'labels', labels)
        #acc_epoc.append([epoch,np.mean(np.array(torch.argmax(input=outputs, dim=1)) == np.array(labels))])
        acc_epoc.append([epoch,running_loss])
        outputs_test = net(test_input, vocabulary, dataset_lengths)
        loss = F.cross_entropy(outputs_test, test_output)
        # calculating perplexity
        perplexity = torch.exp(loss)
        acc_epoc_test.append([epoch,perplexity])
        #acc_epoc = acc_epoc.append([epoch, np.mean(np.array(torch.argmax(input=outputs, dim=1)) == np.array(labels))])

    print('Finished Training')
    print('acc', acc_epoc)
    torch.save(net.state_dict(), dir_input+"outputs/"+NetName().__class__.__name__+".pth")
    return pd.DataFrame(data = acc_epoc, columns = ['epoch','perplexity']) , \
           pd.DataFrame(data = acc_epoc_test, columns = ['epoch','perplexity'])
