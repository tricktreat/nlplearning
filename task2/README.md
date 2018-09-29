### 任务二：基于词嵌入的文本分类

在任务一中，我使用了BoW和N-gram两种方法对文本进行了特征表示，把文本映射到向量空间，再由softmax regression进行分类。这两种特征表示的方法在根本上都是基于单词或者n-gram的出现频数，都需要先构造出字典，再映射文本。这样的方法构造出的特征向量十分的稀疏，并且难以反应上下文单词的联系（N-gram进行改进了）。针对以上的缺点，学者们提出了另外一套方法，利用神经网络来构造词嵌入，不再是直接对文本中单词进行统计分析，而是基于一种神经网络预测模型。

下面我将基于词嵌入的文本向量表示和CNN、RNN的分类方法对任务一中的问题进行改进。

1. 基于词嵌入对文档进行向量表示。分成以下两种方法：

   1. word2vec。

      加载数据集文件，提取文本与分类标签。

      ```python
      def load_data(file_name):
      	texts=[]
      	labels=[]
      	data=pd.read_csv(file_name,sep='\t',header=0,index_col=0)
      	for item in data[['Phrase','Sentiment']].values:
      		texts.append(item[0])
      		labels.append(item[1])
      	return texts,labels
      ```

      加载预训练的word2vec，按“单词:词向量”存储在word2vec字典里。预训练的word2vec文件每行存储一个单词及对应的词向量，词向量的维度是300。

      ```python
      def get_word2vec(file_name):
      	word2vec={}
      	for line in open(file_name,encoding="utf8"):
      		item=line.split()
      		word2vec[item[0]]=np.asarray(item[1:],dtype=np.float32)
      	return word2vec
      ```

      将预料库中的单词根据词频编号，并将语料库转化为编号序列。对序列进行padding操作，即对长度不足maxlen的序列在序列前填充0，对长度超过maxlen的序列进行截断处理。按单词编号把语料中对应单词的词向量保存在word_embedding中，遍历语料库中所有句子，将句子中的单词编号用对应的词向量替换，至此数据的准备工作完成。最后按照4:1切分训练集和测试集，代码如下：

      ```python
      def get_features(texts,word2vec):
      	token=text.Tokenizer()
      	token.fit_on_texts(texts)
      	word_embedding=np.zeros(shape=(len(token.word_index)+1,300))
      	for k,v in token.word_index.items():
      		if word2vec.get(k) is not None:
      			word_embedding[v]= word2vec.get(k)
      	texts_index = sequence.pad_sequences(token.texts_to_sequences(texts), maxlen=40)
      	features=[]
      	for txt in texts_index:
      		feature=[]
      		for i in txt:
      			feature.append(word_embedding[i])
      		features.append(feature)
      	return features

      def get_train_test_set():
      	word2vec=get_word2vec('wordembedding/wiki-news-300d-1M.vec')
      	texts,labels=load_data('data/train.tsv')
      	features=get_features(texts,word2vec)
      	return train_test_split(features, labels, test_size=0.2, shuffle=12)
      ```

   2. glove。

      使用glove数据集将语料库的文本转化为词向量形式，与使用word2vec相似。代码如下：

      ```python
      from keras.preprocessing import text,sequence
      import numpy as np
      import pandas as pd
      from sklearn.model_selection import train_test_split
      import pickle

      def get_glove(file_name):
      	word2vec={}
      	for line in open(file_name,encoding="utf8"):
      		item=line.split()
      		word2vec[item[0]]=np.asarray(item[1:],dtype=np.float32)
      	return word2vec

      def load_data(file_name):
      	texts=[]
      	labels=[]
      	data=pd.read_csv(file_name,sep='\t',header=0,index_col=0)
      	for item in data[['Phrase','Sentiment']].values:
      		texts.append(item[0])
      		labels.append(item[1])
      	return texts,labels

      def get_features(texts,word2vec):
      	token=text.Tokenizer()
      	token.fit_on_texts(texts)
      	word_embedding=np.zeros(shape=(len(token.word_index)+1,300))
      	for k,v in token.word_index.items():
      		if word2vec.get(k) is not None:
      			word_embedding[v]= word2vec.get(k)
      	texts_index = sequence.pad_sequences(token.texts_to_sequences(texts), maxlen=40)
      	features=[]
      	for txt in texts_index:
      		feature=[]
      		for i in txt:
      			feature.append(word_embedding[i])
      		features.append(feature)
      	return features

      def get_train_test_set():
      	word2vec=get_glove('wordembedding/glove.6B.300d.txt')
      	texts,labels=load_data('data/train.tsv')
      	features=get_features(texts,word2vec)
      	return train_test_split(features, labels, test_size=0.2, shuffle=12)
      ```

2. 构建分类模型，使用PyTorch实现。分成以下两种方法：

   1. TextCNN文本分类模型。

      卷积神经网络一般用于图像的特征提取，因为它能够在空间上利用图像的局部特征。可以把文本的看成是空间上的信息分布，将卷积神经网络应用于文本，也能够提取文本的上下文特征，进而进行文本的分类。

      首先定义TextCNN模型，继承自nn.Module，在\_\_init\_\_()中引入模型要使用到的神经网络层，包括三个卷积层和一个全连接层，每个卷积层由三个组件叠加，包括卷积层、非线性激活层和最大池化层，三层卷积层的卷积核尺寸分别为7\*7/5\*5/3\*3，步长为(1,3)，三层池化层的核尺寸分别为5\*5/3\*3/2\*2，卷积层、池化层输入输出的尺寸存在如下关系：
      $$
      \begin{align}\begin{aligned}H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] - \text{dilation}[0]
                \times (\text{kernel_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor\\W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] - \text{dilation}[1]
                \times (\text{kernel_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor\end{aligned}\end{align}
      $$
      根据上式，计算出经过三层卷积和池化后的Tensor尺寸为32(通道)\*21(宽度)*23(长度)，因此定义全连接层的输入神经元数目为15456，输出神经元数目为类别数目5。

      在forward函数中，连接各层，有两处需要注意：第一，卷积层的输入数据需要四个维度，分别是batch_size/channel_num/height/width，所以需要将传入forward的输入数据调整维度。第二在最后一个卷积层和全连接层之间，需要将卷积层的输出数据展开成一维向量，再作为输入传入全连接层。

      ```python
      class TextCNN(nn.Module):
          def __init__(self):
              super(TextCNN, self).__init__()
              self.conv1 = nn.Sequential(
                  nn.Conv2d(1, 8, kernel_size=7, stride=(1,3)),
                  nn.ReLU(),
                  nn.MaxPool2d(kernel_size=5, stride=(1,3))
              )

              self.conv2 = nn.Sequential(
                  nn.Conv2d(8, 16, kernel_size=5, stride=1),
                  nn.ReLU(),
                  nn.MaxPool2d(kernel_size=3, stride=1)
              )

              self.conv3 = nn.Sequential(
                  nn.Conv2d(16, 32, kernel_size=3, stride=1),
                  nn.ReLU(),
                  nn.MaxPool2d(kernel_size=2, stride=1)
              )

              self.fc = nn.Linear(15456, 5)

          def forward(self, x_input):
              x_input = x_input.view(-1,1,40,300)
              x = self.conv1(x_input)
              x = self.conv2(x)
              x = self.conv3(x)
              x = x.view(x.size(0), -1)
              #print(x.shape)
              x = self.fc(x)
              return x
      ```

      模型定义好之后，进行训练：首先载入训练数据（glove或者word2vec）、设置超参数、实例化模型和优化器。这里EPOCH设置为10，batch_size设置为100，学习率设置为0.01,优化器采用adam，损失函数为交叉熵损失函数。每一个batch_size，在训练集上输入loss和accuracy。代码如下：

      ```python
      x_train, x_test, y_train, y_test=get_train_test_set()

      EPOCH=10

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
      ```

      训练结束后，在测试集上对模型进行评测：

      ```
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
      ```

   2. TextLSTM文本分类模型。


      循环神经网络是一类用于处理序列数据的神经网络，在处理当前时间点的输入时，它能够有效利用之前时间点上的信息。而LSTM在RNN的基础上解决了长程依赖的问题，使得网络模型能够利用更久远的历史信息。文本在时间上是一种典型的序列数据，文本的上下文在时间维度上分布。所以，我们能够使用LSTM网络提取文本特征从而进行文本分类。
    
      首先定义TextLSTM模型，模型由一个LSTM网络和一个全连接层组成。有两点需要注意：第一，LSTM的初始状态需要设置初始值，这里设置成零向量。第二，lstm的输入为四个维度的Tensor，意义分别是句子序列长度、batch_size和单词词向量的维度。
    
      ```python
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
      ```
    
      训练和评估过程同TextCNN，代码如下：
    
      ```python
      x_train, x_test, y_train, y_test=get_train_test_set()
      EPOCH=10
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
      ```

3. 实验结果：

   组合步骤1（基于词嵌入对文档进行向量表示）和步骤2（构建分类模型），我一共做了四组实验，分别是word2vec+TextRNN、glove+TextRNN、word2vec+TextLSTM和glove+TextLSTM。由于计算机性能问题，实验中为了减少计算量，batch_size设置为100，epoch设置为10次。实验结果如下：

   |  #   | 词向量类型 | 模型选择 |                accuracy                |
   | :--: | :--------: | :------: | :------------------------------------: |
   |  1   |  word2vec  | TextCNN  | 0.5791（训练了6个epoch，电脑内存不足） |
   |  2   |   glove    | TextCNN  | 0.5652（训练了7个epoch，电脑内存不足） |
   |  3   |  word2vec  | TextLSTM | 0.5048（训练了6个epoch，电脑内存不足） |
   |  4   |   glove    | TextLSTM | 0.4902（训练了6个epoch，电脑内存不足） |

   结果表明word2vec+TextCNN的训练效果更好，但是由于机器性能的限制，训练并不彻底，由此得出的实验结果可靠性不足。