import sys

import chainer as chainer
import torch

########################################################################################################################
# Basic RNN language model
########################################################################################################################

class BasicRNNLM(torch.nn.Module):

    def __init__(self,vocabulary_size,with_drops,method):
        super(BasicRNNLM,self).__init__()

        # Configuration of our model
        self.num_layers=2
        embedding_size=200
        self.hidden_size=200
        dropout_prob=0.5
        self.with_drops = with_drops
        self.method = method
        # Define embedding layer
        self.embed=torch.nn.Embedding(vocabulary_size,embedding_size)

        # Define LSTM
        self.lstm=torch.nn.LSTM(embedding_size,self.hidden_size,self.num_layers,dropout=dropout_prob,batch_first=True)

        # Define GRU
        self.gru=torch.nn.GRU(embedding_size,self.hidden_size,self.num_layers,dropout=dropout_prob,batch_first=True)


        # Define dropout
        self.drop=torch.nn.Dropout(dropout_prob)

        # Define output layer
        self.fc=torch.nn.Linear(self.hidden_size,vocabulary_size)

        # Init weights
        init_range=0.1
        self.embed.weight.data.uniform_(-init_range,init_range)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-init_range,init_range)

        return

    def forward(self,x,h):
        # Apply embedding (encoding)
        y=self.embed(x)
        # Run LSTM
        if self.with_drops == True:
            y=self.drop(y)
        if self.method == 'LSTM':
            y,h=self.lstm(y,h)
        else:
            y,h=self.gru(y,h)
        if self.with_drops == True:
            y=self.drop(y)
        # Reshape
        y=y.contiguous().reshape(-1,self.hidden_size)
        # Fully-connected (decoding)
        y=self.fc(y)
        # Return prediction and states
        return y,h

    def get_initial_states(self,batch_size):
        # Set initial hidden and memory states to 0
        return (torch.autograd.Variable(torch.zeros(self.num_layers,batch_size,self.hidden_size))#.cuda(),
                ,torch.autograd.Variable(torch.zeros(self.num_layers,batch_size,self.hidden_size)) )#.cuda())

    def detach(self,h):
        # Detach returns a new variable, decoupled from the current computation graph
        return h[0].detach(),h[1].detach()



import sys,argparse
import numpy as np
import  pickle
from tqdm import tqdm

"""""
dir_input = r"C:\\Users\\Bengal\Downloads\PTB"+"\\"
train = open(dir_input+'ptb.train.txt').read()
valid = open(dir_input+'ptb.valid.txt').read()
test = open(dir_input+'ptb.test.txt').read()
"""""
import chainer
import numpy as np

dir_input = r"C:\\Users\\Bengal\Downloads\PTB"+"\\"
train = open(dir_input+'ptb.train.txt').read().replace('\n','<eos>').split(' ')
valid = open(dir_input+'ptb.valid.txt').read().replace('\n','<eos>').split(' ')
test = open(dir_input+'ptb.test.txt').read().replace('\n','<eos>').split(' ')

def Vocab(sentence, Vocabulary):
    for word in sentence:
        if word not in Vocabulary:
            Vocabulary[word] = len(Vocabulary)
    return Vocabulary

ptb_dict = Vocab(train,{})
ptb_dict = Vocab(valid,ptb_dict)
ptb_dict = Vocab(test,ptb_dict)

def replace_words_nums(sentence):
    sentence_tmp = []
    for word in sentence:
        sentence_tmp.append(ptb_dict[word])
    return sentence_tmp

train = np.array(replace_words_nums(train))
val = np.array(replace_words_nums(valid))
test = np.array(replace_words_nums(test))

#ptb_dict = chainer.datasets.get_ptb_words_vocabulary()
#train, val, test = chainer.datasets.get_ptb_words()

vocabulary_size =len(ptb_dict)
# Make it pytorch
data_train=torch.LongTensor(train.astype(np.int64))
data_valid=torch.LongTensor(val.astype(np.int64))
data_test=torch.LongTensor(test.astype(np.int64))

# Make batches
batch_size = 35
num_batches=data_train.size(0)//batch_size         # Get number of batches
data_train=data_train[:num_batches*batch_size]     # Trim last elements
data_train=data_train.view(batch_size,-1)          # Reshape
num_batches=data_valid.size(0)//batch_size
data_valid=data_valid[:num_batches*batch_size]
data_valid=data_valid.view(batch_size,-1)
num_batches=data_test.size(0)//batch_size
data_test=data_test[:num_batches*batch_size]
data_test=data_test.view(batch_size,-1)

########################################################################################################################
# Inits
########################################################################################################################

print ('Init...')
learning_rate = 0.1
# Instantiate and init the model, and move it to the GPU
model_LSTM  =BasicRNNLM(vocabulary_size=vocabulary_size,with_drops=False,method='LSTM')#.cuda()
model_LSTM_drop  =BasicRNNLM(vocabulary_size=vocabulary_size,with_drops=True,method='LSTM')#.cuda()
model_GRU  =BasicRNNLM(vocabulary_size=vocabulary_size,with_drops=False,method='GRU')#.cuda()
model_GRU_drop  =BasicRNNLM(vocabulary_size=vocabulary_size,with_drops=True,method='GRU')#.cuda()

# Define loss function
criterion=torch.nn.CrossEntropyLoss(size_average=False)

# Define optimizer
optimizer=torch.optim.SGD(model_LSTM.parameters(),lr=learning_rate)



########################################################################################################################
# Train/test routines
########################################################################################################################
bptt = 35
def train(data,model,criterion,optimizer):
    clip_norm = 0.25
    # Set model to training mode (we're using dropout)
    model.train()
    # Get initial hidden and memory states
    states=model.get_initial_states(data.size(0))
    #train_loss = []
    # Loop sequence length (train)
    for i in tqdm(range(0,data.size(1)-1,bptt),desc='> Train',ncols=100,ascii=True):

        # Get the chunk and wrap the variables into the gradient propagation chain + move them to the GPU
        seqlen=int(np.min([bptt,data.size(1)-1-i]))
        x=torch.autograd.Variable(data[:,i:i+seqlen])#.cuda()
        y=torch.autograd.Variable(data[:,i+1:i+seqlen+1])#.cuda()

        # Truncated backpropagation
        states=model.detach(states)     # Otherwise the model would try to backprop all the way to the start of the data set

        # Forward pass
        logits,states=model.forward(x,states)
        loss=criterion(logits,y.reshape(-1))
        #train_loss.append(np.exp(loss))
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(),clip_norm)
        optimizer.step()

    return model #,train_loss


def eval(data,model,criterion):

    # Set model to evaluation mode (we're using dropout)
    model.eval()
    # Get initial hidden and memory states
    states=model.get_initial_states(data.size(0))

    # Loop sequence length (validation)
    total_loss=0
    num_loss=0
    for i in tqdm(range(0,data.size(1)-1,bptt),desc='> Eval',ncols=100,ascii=True):

        # Get the chunk and wrap the variables into the gradient propagation chain + move them to the GPU
        seqlen=int(np.min([bptt,data.size(1)-1-i]))
        x=torch.autograd.Variable(data[:,i:i+seqlen],volatile=True)#.cuda()
        y=torch.autograd.Variable(data[:,i+1:i+seqlen+1],volatile=True)#.cuda()

        # Truncated backpropagation
        states=model.detach(states)     # Otherwise the model would try to backprop all the way to the start of the data set

        # Forward pass
        logits,states=model.forward(x,states)
        loss=criterion(logits,y.reshape(-1))

        # Log stuff
        total_loss+=loss.data.cpu().numpy()
        num_loss+=np.prod(y.size())

    return float(total_loss)/float(num_loss)



########################################################################################################################
# Train/validation/test
########################################################################################################################

def run_dataset_in_net(model_input,learning_rate = learning_rate, data_train = data_train, data_valid = data_valid, data_test = data_test, optimizer = optimizer):
    print('Train...')
    num_epochs = 1
    anneal_factor = 2.0
    # Loop training epochs
    lr=learning_rate
    best_val_loss=np.inf
    perplexity_train = []
    perplexity_val = []
    perplexity_test = []
    for e in tqdm(range(num_epochs),desc='Epoch',ncols=100,ascii=True):

        # Train
        model =train(data_train,model_input,criterion,optimizer)
        train_loss = eval(data_train, model, criterion)

        # Validation
        val_loss=eval(data_valid,model,criterion)

        # Anneal learning rate
        if val_loss<best_val_loss:
            best_val_loss=val_loss
        else:
            lr/=anneal_factor
            optimizer=torch.optim.SGD(model.parameters(),lr=lr)

        # Test
        test_loss=eval(data_test,model,criterion)

        # Report
        msg='Epoch %d: \tValid loss=%.4f \tTest loss=%.4f \tTest perplexity=%.1f'%(e+1,val_loss,test_loss,np.exp(test_loss))
        perplexity_train.append(np.exp(train_loss))
        perplexity_val.append(np.exp(val_loss))
        perplexity_test.append(np.exp(test_loss))
        tqdm.write(msg)

        return perplexity_train,perplexity_val, perplexity_test


train, val, test = run_dataset_in_net(model_input = model_LSTM)#,learning_rate = learning_rate, data_train = data_train, data_valid = data_valid, data_test = data_test, optimizer = optimizer):


"""
########################################################

# Parse arguments
parser=argparse.ArgumentParser(description='Main script using CIFAR-10')
parser.add_argument('--seed',default=333,type=int,required=False,help='(default=%(default)d)')
parser.add_argument('--filename_in',default='../dat/ptb.pkl',type=str,required=False,help='(default=%(default)s)')
parser.add_argument('--batch_size',default=20,type=int,required=False,help='(default=%(default)d)')
parser.add_argument('--num_epochs',default=40,type=int,required=False,help='(default=%(default)d)')
parser.add_argument('--bptt',default=35,type=int,required=False,help='(default=%(default)d)')
parser.add_argument('--learning_rate',default=20,type=float,required=False,help='(default=%(default)f)')
parser.add_argument('--clip_norm',default=0.25,type=float,required=False,help='(default=%(default)f)')
parser.add_argument('--anneal_factor',default=2.0,type=float,required=False,help='(default=%(default)f)')
args=parser.parse_args()
print('*'*100,'\n',args,'\n','*'*100)

# Import pytorch stuff
import torch
import torchvision

# Set random seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)
else: print ('[CUDA unavailable]'); sys.exit()

# Model import
#import ptb_model

########################################################################################################################
# Load data
########################################################################################################################

print('Load data...')

# Load numpy data
data_train,data_valid,data_test,vocabulary_size=pickle.load(open(args.filename_in,'rb'))

# Make it pytorch
data_train=torch.LongTensor(data_train.astype(np.int64))
data_valid=torch.LongTensor(data_valid.astype(np.int64))
data_test=torch.LongTensor(data_test.astype(np.int64))

# Make batches
num_batches=data_train.size(0)//args.batch_size         # Get number of batches
data_train=data_train[:num_batches*args.batch_size]     # Trim last elements
data_train=data_train.view(args.batch_size,-1)          # Reshape
num_batches=data_valid.size(0)//args.batch_size
data_valid=data_valid[:num_batches*args.batch_size]
data_valid=data_valid.view(args.batch_size,-1)
num_batches=data_test.size(0)//args.batch_size
data_test=data_test[:num_batches*args.batch_size]
data_test=data_test.view(args.batch_size,-1)


########################################################################################################################
# Inits
########################################################################################################################

print 'Init...'

# Instantiate and init the model, and move it to the GPU
model=BasicRNNLM(vocabulary_size)#.cuda()

# Define loss function
criterion=torch.nn.CrossEntropyLoss(size_average=False)

# Define optimizer
optimizer=torch.optim.SGD(model.parameters(),lr=args.learning_rate)

########################################################################################################################
# Train/test routines
########################################################################################################################

def train(data,model,criterion,optimizer):

    # Set model to training mode (we're using dropout)
    model.train()
    # Get initial hidden and memory states
    states=model.get_initial_states(data.size(0))

    # Loop sequence length (train)
    for i in tqdm(range(0,data.size(1)-1,args.bptt),desc='> Train',ncols=100,ascii=True):

        # Get the chunk and wrap the variables into the gradient propagation chain + move them to the GPU
        seqlen=int(np.min([args.bptt,data.size(1)-1-i]))
        x=torch.autograd.Variable(data[:,i:i+seqlen]).cuda()
        y=torch.autograd.Variable(data[:,i+1:i+seqlen+1]).cuda()

        # Truncated backpropagation
        states=model.detach(states)     # Otherwise the model would try to backprop all the way to the start of the data set

        # Forward pass
        logits,states=model.forward(x,states)
        loss=criterion(logits,y.view(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(),args.clip_norm)
        optimizer.step()

    return model


def eval(data,model,criterion):

    # Set model to evaluation mode (we're using dropout)
    model.eval()
    # Get initial hidden and memory states
    states=model.get_initial_states(data.size(0))

    # Loop sequence length (validation)
    total_loss=0
    num_loss=0
    for i in tqdm(range(0,data.size(1)-1,args.bptt),desc='> Eval',ncols=100,ascii=True):

        # Get the chunk and wrap the variables into the gradient propagation chain + move them to the GPU
        seqlen=int(np.min([args.bptt,data.size(1)-1-i]))
        x=torch.autograd.Variable(data[:,i:i+seqlen],volatile=True).cuda()
        y=torch.autograd.Variable(data[:,i+1:i+seqlen+1],volatile=True).cuda()

        # Truncated backpropagation
        states=model.detach(states)     # Otherwise the model would try to backprop all the way to the start of the data set

        # Forward pass
        logits,states=model.forward(x,states)
        loss=criterion(logits,y.view(-1))

        # Log stuff
        total_loss+=loss.data.cpu().numpy()
        num_loss+=np.prod(y.size())

    return float(total_loss)/float(num_loss)

########################################################################################################################
# Train/validation/test
########################################################################################################################

print 'Train...'

# Loop training epochs
lr=args.learning_rate
best_val_loss=np.inf
for e in tqdm(range(args.num_epochs),desc='Epoch',ncols=100,ascii=True):

    # Train
    model=train(data_train,model,criterion,optimizer)

    # Validation
    val_loss=eval(data_valid,model,criterion)

    # Anneal learning rate
    if val_loss<best_val_loss:
        best_val_loss=val_loss
    else:
        lr/=args.anneal_factor
        optimizer=torch.optim.SGD(model.parameters(),lr=lr)

    # Test
    test_loss=eval(data_test,model,criterion)

    # Report
    msg='Epoch %d: \tValid loss=%.4f \tTest loss=%.4f \tTest perplexity=%.1f'%(e+1,val_loss,test_loss,np.exp(test_loss))
    tqdm.write(msg)
"""