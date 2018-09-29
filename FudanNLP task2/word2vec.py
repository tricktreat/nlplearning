from keras.preprocessing import text,sequence
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

def get_word2vec(file_name):
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
	word2vec=get_word2vec('wordembedding/wiki-news-300d-1M.vec')
	texts,labels=load_data('data/train.tsv')
	features=get_features(texts,word2vec)
	return train_test_split(features, labels, test_size=0.2, shuffle=12)