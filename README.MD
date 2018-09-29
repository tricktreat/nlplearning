## NLP入门小实验

### 任务一：基于机器学习的文本分类

文本分类是常见的自然语言处理问题，目标是把文本分类到已经定义好的类别中去，常见的应用有垃圾-非垃圾邮件分类问题，评价文本的情感分类，新闻文本的主题分类等。

基于机器学习的文本分类主要有以下步骤：

1. 准备数据集：从[Classify the sentiment of sentences from the Rotten Tomatoes dataset](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews)下载任务所需数据，部分数据如下。数据集标注了每一个Phrase的Sentiment，Sentiment包括了以下5类：

   + 0 - negative

   + 1 - somewhat negative

   + 2 - neutral

   + 3 - somewhat positive

   + 4 - positive

     | PhraseId | SentenceId | Phrase                                                       | Sentiment |
     | -------- | ---------- | ------------------------------------------------------------ | --------- |
     | 1        | 1          | A series of escapades demonstrating the adage that what is good for the goose is also good for the gander , some of which occasionally amuses but none of which amounts to much of a story . | 1         |
     | 2        | 1          | A series of escapades demonstrating the adage that what is good for the goose | 2         |
     | 3        | 1          | A series                                                     | 2         |
     | 4        | 1          | A                                                            | 2         |
     | 5        | 1          | series                                                       | 2         |
     | 6        | 1          | of escapades demonstrating the adage that what is good for the goose | 2         |
     | 7        | 1          | of                                                           | 2         |
     | 8        | 1          | escapades demonstrating the adage that what is good for the goose | 2         |
     | 9        | 1          | escapades                                                    | 2         |


   1. 导入数据：

      从tsv文件中导入所有训练数据，提取出所需要的Phrase文本数据和Sentiment标签数据，构建ndarray对象。

      ```python
      def load_data(file_name):
          data=pd.read_csv(file_name,sep='\t',index_col=0,header=0);
          return data[['Phrase','Sentiment']].values
      ```

   2. 划分训练集/验证集/测试集

      数据集的数据量为十万级别，对数据集按照训练集：验证集：测试集=6：2：2进行划分，在划分之前数据集要进行shuffle。

      ```python
      def preprocess(x,y):
          random.seed(1)
          random.shuffle(x)
          random.seed(1)
          random.shuffle(y)
          n=len(y)
          train_ratio,val_ratio,test_ratio=0.6,0.2,0.2
          train_size,val_size,test_size=int(n*train_ratio),int(n*val_ratio),int(n*test_ratio)
          train_x,train_y=x[:train_size],y[:train_size]
          val_x,val_y=x[train_size:-test_size], y[train_size:-test_size]
          test_X,test_y=x[-test_size:],y[-test_size:]

      ```

3. 特征表示

   机器学习算法难以直接利用原始的文本信息，所以需要对文本进行数值（向量）表示。将原始数据将被转换为向量，即把文本映射到向量空间，构造的方法一般分为两大类：计数模型和预测模型。本次任务主要是运用计数模型对文本进行向量表示，一般有BoW、N-gram等方法。

   1. BoW(Bag-of-Word)：词袋模型是一种基于文档中单词出现的“某种度量”的文本表示方法，它需要构建已知单词的字典表和确定单词出现的度量标准。单词出现的度量标准最简单的就是文档中单词出现的频数，另外还有出现-未出现布尔值度量和TF-IDF向量。
   2. N-gram：在BoW的基础上进行了改进，关联了单词的context信息，有更强的文本特征表达能力，一般使用bigram，即将相邻的单词组合，成为构建的字典表中的一项。

   针对以上描述，特征工程分为以下两个步骤：

   1. 设计字典表。

      对于BoW方法：这里对文本中出现的所有单词按照出现的频数排序，排序后的索引作为对应单词的编号。处理文本时，先去除信息量较少的停用词和标点符号。

      单词频数统计结果（top 10）：[('film', 6733), ('movie', 6241), ('n', 4025), ('one', 3784), ('like', 3190), ('story', 2539), ('rrb', 2438), ('good', 2261), ('lrb', 2098), ('time', 1919)]，对应字典表（top 10）：['film': 0, 'movie': 1, 'n': 2, 'one': 3, 'like': 4, 'story':5, 'rrb': 6, 'good': 7, 'lrb': 8, 'time': 9]

      ```python
      def build_dict(data):
          counter=collections.Counter()
          for item in data:
              for word in re.split(' +',re.sub(r'[{}]+'.format(punctuation),' ',item[0])):
                  counter[word.lower()]+=1
          del counter['']
          for i in sw:
               del counter[i]
          sorted_word_to_cnt=sorted(counter.items(),key=itemgetter(1),reverse=True)
          words=[x[0] for x in sorted_word_to_cnt]
          word_id_dic={k:v for (k,v) in zip(words[:FEATURE_NUM],range(FEATURE_NUM))}
          return word_id_dic
      ```

      对于N-gram方法：这里N=2，先对文本中相邻的两个单词进行切割视为一个单词，再同BoW方法统计单词组合的频数、排序、编号。

   2. 创建文本向量。

      在上个步骤中已经得到了根据单词或者bigram的频数确定的字典表，选择出现最频繁的FEATURE_NUM个单词作为文本特征。对数据集中的每一个文本进行转化，创建特征向量。文本的特征向量包含FEATURE_NUM个元素，每个元素对应字典表中相同位置的单词或bigram，元素为该单词或bigram在该文本中出现的频数。

      文本特征向量示例："A series of escapades demonstrating the adage that what is good for the goose is also good for the gander , some of which occasionally amuses but none of which amounts to much of a story ."------->  [ 0, 0, 0, 0, 0, 1, 0, 2,  0, 0, ··· 0]（字典表前十个元素为['film': 0, 'movie': 1, 'n': 2, 'one': 3, 'like': 4, 'story':5, 'rrb': 6, 'good': 7, 'lrb': 8, 'time': 9]）

      ```python
      def construct_data(data,word_id_dic):
          labels=[i[1] for i in data]
          features=[]
          for item in data:
              features_item=np.zeros(FEATURE_NUM)
              for word in re.split(' +',re.sub(r'[{}]+'.format(punctuation),' ',item[0])):
                  word=word.lower()
                  if word_id_dic.get(word)!=None:
                      features_item[word_id_dic[word]]+=1
              features.append(features_item)
          return features,labels
      ```

3. 建立模型。

   最后一步是利用之前创建的特征训练一个分类器，机器学习中有很多分类模型可供选择，例如朴素贝叶斯、线性分类器、支持向量机、深度神经网络等。本次任务我使用了线性分类器中的softmax regression。

   softmax是logisitic regression在多分类问题上的推广，我们要拟合的目标变量，是一个one-hot vector（只有一个1，其余均为0）。定义如下：

   - $x$为输入向量，$d*1$列向量，$d$为特征数量。$y$为label，即要拟合的目标，$1*c$行向量（one-hot vector），$c$为类别数。$\hat y$为输出值（预测分类，归一化），形状同label。
   - $W$为权重矩阵，形状为$c*d$，$b$为每个类别对应超平面的偏置项，$b$的形状为$1*c$。
   - $z=Wx+b$：线性分类器输出，$c*1$列向量

   它们的关系为：
   $$
   \left\{\begin{aligned}&z=Wx+b\\& \hat{y}=\mathrm{softmax}(z)=\frac{exp(z)}{\sum exp(z)} \end{aligned}\right.
   $$
   选择交叉熵函数作为loss function，对于一个训练样本：
   $$
   CE(z) = -y^Tlog(\hat{y})
   $$
   进行随机梯度下降，$\lambda$为学习率：
   $$
   \begin{aligned}&W \leftarrow W - \lambda (\hat{y}-y)\cdot x^T \\&b \leftarrow b - \lambda (\hat{y}-y)\end{aligned}
   $$
   下面进行模型训练，损失函数为交叉熵，学习率$\lambda = 1.0 / (\alpha * (t + t_0))$，$\alpha$为正则项的惩罚系数，最大迭代次数为5，训练结束误差阈值为1e-3，训练前对训练集进行shuffle，选择特征数目为FEATURE_NUM。

   batch训练模型：

   ```python
   def train_model():
   	#batch SGD
       model = SGDClassifier(random_state=1,learning_rate ='optimal',shuffle =True,loss ='log',max_iter=5,tol=1e-3)
       model.fit(train_x, train_y)
       print("using features: {1}, get val mean accuracy: {0}".format(model.score(val_x, val_y),FEATURE_NUM))
       y_pred = model.predict(test_X)
       print(classification_report(test_y, y_pred))
   ```

   mini-batch训练模型：

   ```python
   def train_model():
       #mini-batch SGD
       batch_size=80 #online set to 1
       mini_batchs=[]
       i=0
       while i+batch_size<=train_size:
           mini_batchs.append((x[i:i+batch_size],y[i:i+batch_size]))
           i+=100
       if i<train_size:
           mini_batchs.append((x[i:train_size],y[i:train_size]))
       model = SGDClassifier(random_state=1,learning_rate ='optimal',shuffle =True,loss ='log',max_iter=5,tol=1e-3)
       for batch_x,batch_y in mini_batchs:
           model.partial_fit(batch_x,batch_y,classes=np.unique([0,1,2,3,4]))

       print("using features: {1}, get val mean accuracy: {0}".format(model.score(val_x, val_y),FEATURE_NUM))
       y_pred = model.predict(test_X)
       print(classification_report(test_y, y_pred))
   ```

4. 结果对比：

   |  #   | feature type | FEATURE_NUM |         learning rate         | batch size | accuracy |
   | :--: | :----------: | :---------: | :---------------------------: | :--------: | :------: |
   |  1   |     BoW      |    1000     | $ 1.0 / (\alpha * (t + t_0))$ |    100     |  0.5596  |
   |  2   |     BoW      |    2000     | $ 1.0 / (\alpha * (t + t_0))$ |    100     |  0.5721  |
   |  3   |     BoW      |    3000     | $ 1.0 / (\alpha * (t + t_0))$ |    100     |  0.5767  |
   |  4   |     BoW      |    3000     | $ 1.0 / (\alpha * (t + t_0))$ |   batch    |  0.5686  |
   |  5   |     BoW      |    3000     |             0.01              |    100     |  0.5336  |
   |  6   |     BoW      |    3000     |             0.05              |    100     |  0.5623  |
   |  7   |    bigram    |    3000     | $ 1.0 / (\alpha * (t + t_0))$ |   batch    |  0.5191  |
   |  8   |    bigram    |    3000     | $ 1.0 / (\alpha * (t + t_0))$ |    100     |  0.5183  |
   |  9   |   trigram    |    3000     | $ 1.0 / (\alpha * (t + t_0))$ |    100     |  0.5147  |
   |  10  |    bigram    |    1000     | $ 1.0 / (\alpha * (t + t_0))$ |    100     |  0.5153  |

	在所有的实验中模型3效果最好，模型参数：Bag-of-Word方法（统计词频数）， FEATURE_NUM为3000，学习率选择$ 1.0 / (\alpha * (t + t_0))$，采用mini-batch（batch-size为100）。将模型3得到的结果提交到kaggle平台，得到的Score为0.59038。

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