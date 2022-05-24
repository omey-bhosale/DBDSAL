#!/usr/bin/env python
# coding: utf-8

# In[17]:


import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# In[6]:


from nltk.tokenize import sent_tokenize
text="""Hello Mr. Smith, how are you doing today? The weather is great, and city is awesome
The sky is pinkish-blue. You shouldn't eat cardboard"""
tokenized_text=sent_tokenize(text)
print(tokenized_text)


# In[7]:


from nltk.tokenize import word_tokenize
tokenized_word=word_tokenize(text)
print(tokenized_word)


# In[8]:


from nltk.probability import FreqDist
fdist = FreqDist(tokenized_word)
print(fdist)


# In[9]:


fdist.most_common(2)


# In[12]:


from nltk.corpus import stopwords
stop_words=set(stopwords.words("english"))
print(stop_words)


# In[14]:


filtered_sent=[]
for w in tokenized_word:
    if w not in stop_words:
        filtered_sent.append(w)
print("Tokenized Sentence:",tokenized_word)
print("Filterd Sentence:",filtered_sent)


# In[15]:


from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
ps = PorterStemmer()
stemmed_words=[]
for w in filtered_sent:
    stemmed_words.append(ps.stem(w))
print("Filtered Sentence:",filtered_sent)
print("Stemmed Sentence:",stemmed_words)


# In[18]:


from nltk.stem.wordnet import WordNetLemmatizer
lem = WordNetLemmatizer()
from nltk.stem.porter import PorterStemmer
stem = PorterStemmer()
word = "flying"
print("Lemmatized Word:",lem.lemmatize(word,"v"))
print("Stemmed Word:",stem.stem(word))


# In[ ]:




