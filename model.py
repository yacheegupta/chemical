#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import sklearn
from sklearn.manifold import TSNE
import gensim
import matplotlib.pyplot as plt


# In[2]:


df_name = pd.read_csv("name.csv", sep = '/t', header = None)
df_syn = pd.read_csv("synonyms.csv",sep='/t', header = None)


# In[3]:


def token(row,column_name):
    return gensim.utils.simple_preprocess(str(row[column_name]).encode('utf-8'))
#convert a document into a list of tokens.
   
name_data = []
syn_data = []

#for name 
for index, row in df_name.iterrows():
    name_data.append(token(row,0))
    
#for synonym
for index, row in df_syn.iterrows():
    syn_data.append(token(row,0))


# In[4]:


model = gensim.models.Word2Vec(size=150, window=10, min_count=2, sg=1, workers=10)
model.build_vocab(syn_data)  # prepare the model vocabulary


# In[5]:


##Train the model
model.train(sentences=syn_data, total_examples=len(syn_data), epochs=model.iter)


# In[6]:


word_vectors = model.wv
count = 0
for word in word_vectors.vocab:
    if count<200:
        print(word)
        count += 1
    else:
        break


# In[7]:


len(word_vectors.vocab)


# In[8]:


vector = model.wv['benzene']
vector


# In[9]:


#Create a two-dimensional semantic representation of word embeddings using t-Distributed Stochastic Neighbor Embedding (t-SNE).
wanted_words = []
count = 0
for word in word_vectors.vocab:
    if count<500:
        wanted_words.append(word)
        count += 1
    else:
        break
wanted_vocab = dict((k, word_vectors.vocab[k]) for k in wanted_words if k in word_vectors.vocab)
wanted_vocab


# In[10]:


X = model[wanted_vocab] # X is an array of word vectors, each vector containing 150 tokens
tsne_model = TSNE(perplexity=40, n_components=2, init="pca", n_iter=10000, random_state=23)
Y = tsne_model.fit_transform(X)
tsne_model


# In[11]:


#Plot the t-SNE output
fig, ax = plt.subplots(figsize=(20,10))
ax.scatter(Y[:, 0], Y[:, 1])
words = list(wanted_vocab)
for i, word in enumerate(words):
    plt.annotate(word, xy=(Y[i, 0], Y[i, 1]))
ax.set_yticklabels([]) #Hide ticks
ax.set_xticklabels([]) #Hide ticks
plt.show()


# In[12]:


#Playing with the trained Word2Vec model

w1 = "water"
model.wv.most_similar(positive=w1, topn=20)


# In[13]:


#find odd item in a list
model.wv.doesnt_match(["water","benzene","soluble", "acqua"])


# In[14]:


#save the model for futue use
model.wv.save_word2vec_format('model.bin')


# In[ ]:




