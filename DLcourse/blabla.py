import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

########################################################################################################################
# LSTM & GRU models
########################################################################################################################

class GRU(torch.nn.Module):

    def __init__(self,vocabulary_size,with_drops):
        super(GRU,self).__init__()

        # Configuration of our model
        self.num_layers=2
        embedding_size=200
        self.hidden_size=200
        dropout_prob=0.5
        self.with_drops = with_drops
        # Define embedding layer
        self.embed=torch.nn.Embedding(vocabulary_size,embedding_size)

        # Define GRU
        self.gru=torch.nn.GRU(embedding_size,self.hidden_size,self.num_layers,batch_first=True)
        self.gru_drop=torch.nn.GRU(embedding_size,self.hidden_size,self.num_layers,dropout=dropout_prob,batch_first=True)

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
            y, h = self.gru(y, h)
            #y,h=self.gru_drop(y,h)
            y = self.drop(y)
        else:
            y,h=self.gru(y,h)

        # Reshape
        y=y.contiguous().reshape(-1,self.hidden_size)
        # Fully-connected (decoding)
        y=self.fc(y)
        # Return prediction and states
        return y,h

    def get_initial_states(self,batch_size):
        # Set initial hidden and memory states to 0
        return (torch.autograd.Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)) )
             #,torch.autograd.Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)) )

    def detach(self,h):
        # Detach returns a new variable, decoupled from the current computation graph
        h[0].detach() #,h[1].detach()

class LSTM(torch.nn.Module):

    def __init__(self,vocabulary_size,with_drops):
        super(LSTM,self).__init__()

        # Configuration of our model
        self.num_layers=2
        embedding_size=200
        self.hidden_size=200
        dropout_prob=0.5
        self.with_drops = with_drops
        # Define embedding layer
        self.embed=torch.nn.Embedding(vocabulary_size,embedding_size)

        # Define LSTM
        self.lstm=torch.nn.LSTM(embedding_size,self.hidden_size,self.num_layers,batch_first=True)
        self.lstm_drop=torch.nn.LSTM(embedding_size,self.hidden_size,self.num_layers,dropout=dropout_prob,batch_first=True)

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
            y, h = self.lstm(y, h)
            #y,h=self.lstm_drop(y,h)
        else:
            y,h=self.lstm(y,h)

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

###########################################################

# Upload data
dir_input = r"C:\\Users\\Bengal\Downloads\PTB"+"\\"
train = open(dir_input+'ptb.train.txt').read().replace('\n','<eos>').split(' ')
valid = open(dir_input+'ptb.valid.txt').read().replace('\n','<eos>').split(' ')
test = open(dir_input+'ptb.test.txt').read().replace('\n','<eos>').split(' ')

# Function: Create dictionary of sentence
def Vocab(sentence, Vocabulary):
    for word in sentence:
        if word not in Vocabulary:
            Vocabulary[word] = len(Vocabulary)
    return Vocabulary

# Create dictionary from our datasets
ptb_dict = Vocab(train,{})      # first dictionary is empty
ptb_dict = Vocab(valid,ptb_dict)    # update dictionary with validation set
ptb_dict = Vocab(test,ptb_dict)     # update dictionary with test set

# Replace the words in a list to their numeric value by dictionary
def replace_words_nums(sentence, dictionary = ptb_dict):
    sentence_tmp = []
    for word in sentence:
        sentence_tmp.append(dictionary[word])
    return sentence_tmp

# Replace the words in our datasets to their numeric value by our prepared ptb_dictionary
train = np.array(replace_words_nums(train))
val = np.array(replace_words_nums(valid))
test = np.array(replace_words_nums(test))


vocabulary_size =len(ptb_dict)

# convert the datasets to torches
data_train=torch.LongTensor(train.astype(np.int64))
data_valid=torch.LongTensor(val.astype(np.int64))
data_test=torch.LongTensor(test.astype(np.int64))

# Make batches
batch_size = 20
# Train
num_batches=data_train.size(0)//batch_size         # Get number of batches
data_train=data_train[:num_batches*batch_size]     # Trim last elements
data_train=data_train.view(batch_size,-1)          # Reshape
# Validation
num_batches=data_valid.size(0)//batch_size
data_valid=data_valid[:num_batches*batch_size]
data_valid=data_valid.view(batch_size,-1)
# Test
num_batches=data_test.size(0)//batch_size
data_test=data_test[:num_batches*batch_size]
data_test=data_test.view(batch_size,-1)

########################################################################################################################
# Inits
########################################################################################################################


# Instantiate and init the models
# LSTM models
model_LSTM  =LSTM(vocabulary_size=vocabulary_size,with_drops=False)
model_LSTM_drop  =LSTM(vocabulary_size=vocabulary_size,with_drops=True)

# GRU models
model_GRU  =GRU(vocabulary_size=vocabulary_size,with_drops=False)
model_GRU_drop =GRU(vocabulary_size=vocabulary_size,with_drops=True)

########################################################################################################################
# Train/test routines
########################################################################################################################
def train(data,model,criterion,optimizer,bptt = 35):
    clip_norm = 5.0
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

def eval(data,model,criterion,bptt = 35):

    # Set model to evaluation mode (we're using dropout)
    model.eval()
    # Get initial hidden and memory states
    states=model.get_initial_states(data.size(0))

    # Loop sequence length (validation)
    total_loss=0
    num_loss=0
    for i in tqdm(range(0,data.size(1)-1,bptt),desc='> Eval',ncols=100,ascii=True):

        # Get the chunk and wrap the variables into the gradient propagation chain
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

def run_dataset_in_net(model_input,file_name,num_epochs,anneal_factor = 2,dir_input =dir_input, lr = 1, data_train = data_train, data_valid = data_valid, data_test = data_test):
    print('Train...')
    # Define optimizer
    optimizer = torch.optim.SGD(model_LSTM.parameters(), lr=1)

    # Define loss function
    criterion = torch.nn.CrossEntropyLoss(size_average=False)

    # for gradient explotion
    best_val_loss=np.inf

    # empty list for the perplexity losses
    perplexity_train = []
    perplexity_val = []
    perplexity_test = []

    # Loop training epochs
    for e in tqdm(range(num_epochs),desc='Epoch',ncols=100,ascii=True):
        # Train
        model =train(data_train,model_input,criterion,optimizer)
        train_loss = eval(data_train, model, criterion)
        # Validation
        val_loss=eval(data_valid,model,criterion)
        # Test
        test_loss=eval(data_test,model,criterion)

        if val_loss<best_val_loss:
            best_val_loss=val_loss
        else:
            if (e > 5) & (file_name in ('model_GRU_drop','model_LSTM_drop')):
                lr/=anneal_factor
            if (e > 3) & (file_name in ('model_GRU','model_LSTM')):
                lr/=anneal_factor
            optimizer=torch.optim.SGD(model.parameters(),lr=lr)

        # Report
        msg='Epoch %d: \tValid loss=%.4f \tTest loss=%.4f \tTest perplexity=%.1f'%(e+1,val_loss,test_loss,np.exp(test_loss))
        perplexity_train.append(np.exp(train_loss))
        perplexity_val.append(np.exp(val_loss))
        perplexity_test.append(np.exp(test_loss))
        tqdm.write(msg)
    DF = pd.DataFrame({'Train':perplexity_train,'Validation':perplexity_val,'Test':perplexity_test})
    DF.to_csv(dir_input+ "outputs/" + file_name + ".csv")
    torch.save(model.state_dict(), dir_input + "outputs/" + file_name + ".pth")
    return DF

########################################################################################################################
# RUN THE MODELS (run it only if you want to *retrain* the models)
########################################################################################################################
#CHEN RUN
# LSTM = run_dataset_in_net(model_input = model_LSTM,file_name = 'model_LSTM', num_epochs = 13, anneal_factor = 2.0)#,learning_rate = learning_rate, data_train = data_train, data_valid = data_valid, data_test = data_test, optimizer = optimizer):
# LSTM_drop = run_dataset_in_net(model_input = model_LSTM_drop,file_name = 'model_LSTM_drop', num_epochs = 17,anneal_factor = 1.2)#,learning_rate = learning_rate, data_train = data_train, data_valid = data_valid, data_test = data_test, optimizer = optimizer):
# #OMRI RUN
# GRU = run_dataset_in_net(model_input = model_GRU,file_name = 'model_GRU', num_epochs = 13, anneal_factor = 2.0)#,learning_rate = learning_rate, data_train = data_train, data_valid = data_valid, data_test = data_test, optimizer = optimizer):
# GRU_drop = run_dataset_in_net(model_input = model_GRU_drop,file_name = 'model_GRU_drop', num_epochs = 17,anneal_factor = 1.2)#,learning_rate = learning_rate, data_train = data_train, data_valid = data_valid, data_test = data_test, optimizer = optimizer):

########################################################################################################################
#   Upload loss results without running the models
########################################################################################################################
LSTM_DF = pd.read_csv(dir_input+ "outputs/" + "model_LSTM" + ".csv")[['Train','Validation','Test']]
LSTM_drop_DF = pd.read_csv(dir_input+ "outputs/" + "model_LSTM_drop" + ".csv")[['Train','Validation','Test']]
GRU_DF = pd.read_csv(dir_input+ "outputs/" + "model_GRU" + ".csv")[['Train','Validation','Test']]
GRU_drop_DF = pd.read_csv(dir_input+ "outputs/" + "model_GRU_drop" + ".csv")[['Train','Validation','Test']]

# Prepare dataframes of train validation and test performance
trains = pd.DataFrame({'LSTM':LSTM_DF['Train'],'LSTM_Dropout':LSTM_drop_DF['Train'],'GRU':GRU_DF['Train'],'GRU_Dropout':GRU_drop_DF['Train']})
vals = pd.DataFrame({'LSTM':LSTM_DF['Validation'],'LSTM_Dropout':LSTM_drop_DF['Validation'],'GRU':GRU_DF['Validation'],'GRU_Dropout':GRU_drop_DF['Validation']})
tests = pd.DataFrame({'LSTM':LSTM_DF['Test'],'LSTM_Dropout':LSTM_drop_DF['Test'],'GRU':GRU_DF['Test'],'GRU_Dropout':GRU_drop_DF['Test']})

########################################################################################################################
#   Plots
########################################################################################################################
# By model
LSTM_DF.plot(lw=2, colormap='jet', marker='.', markersize=10, title='LSTM perplexity')
LSTM_drop_DF.plot(lw=2, colormap='jet', marker='.', markersize=10, title='LSTM with Dropouts perplexity')
GRU_DF.plot(lw=2, colormap='jet', marker='.', markersize=10, title='GRU perplexity')
GRU_drop_DF.plot(lw=2, colormap='jet', marker='.', markersize=10, title='GRU with Dropouts perplexity')

# By dataset
trains.plot(lw=2, colormap='jet', marker='.', markersize=10, title='Train dataset perplexity')
vals.plot(lw=2, colormap='jet', marker='.', markersize=10, title='Validation dataset perplexity')
tests.plot(lw=2, colormap='jet', marker='.', markersize=10, title='Test dataset perplexity')

########################################################################################################################
# Evaluate a test sentence with the prepared weights
########################################################################################################################

def TestingNet(test_set, NetName,dictionary = ptb_dict,with_drops = False):
    batch_size = 20
    net = NetName(vocabulary_size = len(dictionary),with_drops = with_drops)
    if with_drops == True: drop = '_drop'
    else: drop = ''
    net.load_state_dict(torch.load(dir_input+"outputs\\"+"model_"+net.__class__.__name__+drop+".pth"))

    test_set = test_set.replace('\n','<eos>').split(' ')
    test_set = np.array(replace_words_nums(test_set, dictionary = dictionary))
    test_set = torch.LongTensor(test_set.astype(np.int64))
    num_batches = test_set.size(0) // batch_size
    test_set = test_set[:num_batches * batch_size]
    test_set = test_set.view(batch_size, -1)
    criterion = torch.nn.CrossEntropyLoss(size_average=False)
    loss = eval(test_set, net, criterion)

    return print("Perplexity Loss: ", np.exp(loss))

test_sentence = open(dir_input+'ptb.test.txt').read()
TestingNet(test_set= test_sentence,NetName= LSTM, with_drops = False)

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