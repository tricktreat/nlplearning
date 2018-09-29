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