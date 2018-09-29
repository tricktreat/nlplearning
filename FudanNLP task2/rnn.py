from word2vec import get_train_test_set
import torch
import torch.nn as nn
import torch.nn.functional as F

x_train, x_test, y_train, y_test=get_train_test_set()

EPOCH=10

class TextLSTMC(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, label_size, batch_size):
        super(TextLSTMC, self).__init__()
        self.batch_size=batch_size
        self.hidden_dim=hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2label = nn.Linear(hidden_dim, label_size)
        self.hidden = self.init_hidden()


    def init_hidden(self):
        return (torch.zeros(1, self.batch_size, self.hidden_dim),
                torch.zeros(1, self.batch_size, self.hidden_dim))

    def forward(self, sentence):
        x = sentence.view(40,self.batch_size,-1)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        y  = self.hidden2label(lstm_out[-1])
        return y

batch_size=100
model = TextLSTMC(300,300,5,batch_size)
LR = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_function = nn.CrossEntropyLoss()
test_x=torch.Tensor(x_test)
test_y=torch.LongTensor(y_test)

for epoch in range(EPOCH):
    for i in range(0,(int)(len(x_train)/batch_size)):
        train_x = torch.Tensor(x_train[i*batch_size:i*batch_size+batch_size])
        train_y = torch.LongTensor(y_train[i*batch_size:i*batch_size+batch_size])
        model.hidden = model.init_hidden()
        output = model(train_x)
        loss = loss_function(output, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("### epoch "+str(epoch)+" batch "+str(i)+" ###")
        print("loss: "+str(loss))
        pred_y = torch.max(output, 1)[1].data.squeeze()
        acc = (train_y == pred_y)
        acc = acc.numpy().sum()
        accuracy = acc / (train_y.size(0))
        print("accuracy: "+ str(accuracy))
    

    acc_all = 0
    for i in range(0,(int)(len(test_x)/batch_size)):
        test_output = model(test_x[i*batch_size:i*batch_size+batch_size])
        pred_y = torch.max(test_output, 1)[1].data.squeeze()
        acc = (pred_y == test_y[i*batch_size:i*batch_size+batch_size])
        acc = acc.numpy().sum()
        acc_all = acc_all + acc
    accuracy = acc_all / (test_y.size(0))
    print("###  epoch " + str(epoch) + " ###")
    print("accuracy: " + str(accuracy))