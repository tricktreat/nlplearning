import numpy as np
import pandas as pd
import re
import collections
from operator import itemgetter
from string import punctuation
from nltk.corpus import stopwords
import nltk
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
import csv
from numpy import random

sw = stopwords.words("english")

FEATURE_NUM=1000

def load_data(file_name):
    data=pd.read_csv(file_name,sep='\t',index_col=0,header=0);
    return data[['Phrase','Sentiment']].values

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

def get_result(file_in,file_out,model,word_id_dic):
    data=pd.read_csv(file_in,sep='\t',header=0)
    data=data[['PhraseId','Phrase']].values
    phraseid= [i[0] for i in data]
    features=[]
    for item in data:
        features_item=np.zeros(FEATURE_NUM)
        for word in re.split(' +',re.sub(r'[{}]+'.format(punctuation),' ',item[1])):
            word=word.lower()
            if word_id_dic.get(word)!=None:
                features_item[word_id_dic[word]]+=1
        features.append(features_item)
    y_pred=model.predict(features)
    out = open(file_out,'a', newline='')
    csv_write = csv.writer(out,dialect='excel')
    csv_write.writerow(['PhraseId','Sentiment'])
    for i in zip(phraseid,y_pred):
        csv_write.writerow(list(i))


if __name__ == '__main__':
    data=load_data("data/train.tsv")
    word_id_dic=build_dict(data)
    x,y=construct_data(data,word_id_dic)
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


    # #batch SGD
    # model = SGDClassifier(random_state=1,learning_rate ='constant',eta=0.01,shuffle =True,loss ='log',max_iter=100,tol=1e-3)
    # model.fit(train_x, train_y)

    #mini-batch SGD
    batch_size=100 #online set to 1
    mini_batchs=[]
    i=0
    while i+batch_size<=train_size:
        mini_batchs.append((x[i:i+batch_size],y[i:i+batch_size]))
        i+=batch_size
    if i<train_size:
        mini_batchs.append((x[i:train_size],y[i:train_size]))
    model = SGDClassifier(random_state=1,learning_rate ='optimal',shuffle =True,loss ='log',max_iter=5,tol=1e-3)
    for batch_x,batch_y in mini_batchs:
        model.partial_fit(batch_x,batch_y,classes=np.unique([0,1,2,3,4]))

    print("using features: {1}, get val mean accuracy: {0}".format(model.score(val_x, val_y),FEATURE_NUM))
    y_pred = model.predict(test_X)
    print(classification_report(test_y, y_pred))

    get_result('data/test.tsv','data/submission.csv',model,word_id_dic)

