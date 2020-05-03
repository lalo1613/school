import torch
import torch.nn.functional as F
from DLcourse.RNN import NetLSTM


def Training_LSTM(train_input, train_output, train_lengths, n_epochs=1): # test_input, test_output, ):
    net = NetLSTM()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    acc_epoc = []
    acc_epoc_test = []

    # training net
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        batch_size = 20
        hidden = torch.randn(2,batch_size,200)
        for i in range((len(train_input)//batch_size)+1):

            # inputs = train_input[(batch_size*i):(batch_size*(i+1))]#.apply_(float)
            # input_lengths = train_lengths[(batch_size*i):(batch_size*(i+1))]

            torch.nn.utils.rnn.pack_padded_sequence()

            inputs = train_input[i]
            input_lengths = train_lengths[i]
            outputs = train_output[(batch_size*i):(batch_size*(i+1))]

            optimizer.zero_grad()
            preds, hidden = net(inputs, input_lengths, hidden)
            loss = F.cross_entropy(preds, outputs)
            perplexity = torch.exp(loss)
            print('Loss:', loss, 'PP:', perplexity)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print("loss "+str(running_loss/(i+1)))

        return net

        #print('outputs',torch.argmax(input = outputs, dim = 1), 'labels', labels)
        #acc_epoc.append([epoch,np.mean(np.array(torch.argmax(input=outputs, dim=1)) == np.array(labels))])
        # acc_epoc.append([epoch,running_loss])
        # outputs_test = net(test_input)
        # loss = F.cross_entropy(outputs_test, test_output)
        # # calculating perplexity
        # perplexity = torch.exp(loss)
        # acc_epoc_test.append([epoch,perplexity])
        #acc_epoc = acc_epoc.append([epoch, np.mean(np.array(torch.argmax(input=outputs, dim=1)) == np.array(labels))])
