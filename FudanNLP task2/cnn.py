from word2vec import get_train_test_set
import torch
import torch.nn as nn
import torch.nn.functional as F

x_train, x_test, y_train, y_test=get_train_test_set()

EPOCH=10

class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7, stride=(1,3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5, stride=(1,3))
        )#35 

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1)
        )#32

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1)
        )#29

        self.fc = nn.Linear(15456, 5)

    def forward(self, x_input):
        x_input = x_input.view(-1,1,40,300)
        x = self.conv1(x_input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = TextCNN()
LR = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_function = nn.CrossEntropyLoss()
batch_size=100
test_x=torch.Tensor(x_test)
test_y=torch.LongTensor(y_test)

for epoch in range(EPOCH):
    for i in range(0,(int)(len(x_train)/batch_size)):
        train_x = torch.Tensor(x_train[i*batch_size:i*batch_size+batch_size])
        train_y = torch.LongTensor(y_train[i*batch_size:i*batch_size+batch_size])
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