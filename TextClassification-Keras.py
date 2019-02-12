
# coding: utf-8

# # Text classification with Keras

# ## 1. Load libraries

# In[1]:


import pandas as pd
pd.options.display.max_columns = 200
pd.options.mode.chained_assignment = None

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
from string import punctuation

from collections import Counter
import re
import numpy as np
import seaborn as sns

from wordcloud import WordCloud

import pickle

from tqdm import tqdm_notebook
# tqdm_notebook().pandas()
from tqdm import tqdm
from matplotlib import pyplot as plt
import re
from functools import reduce

get_ipython().magic(u'matplotlib inline')
get_ipython().magic(u"config InlineBackend.figure_format = 'retina'")

from sklearn.model_selection import (train_test_split, KFold, StratifiedKFold,
                                     cross_val_score, GridSearchCV,
                                     learning_curve, validation_curve)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (log_loss, accuracy_score, f1_score,
                             classification_report, confusion_matrix,
                             recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              ExtraTreesClassifier, GradientBoostingClassifier)
from xgboost import XGBClassifier

from nltk.tokenize import WordPunctTokenizer, word_tokenize, sent_tokenize
from bs4 import BeautifulSoup
import unicodedata
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer

import warnings
warnings.filterwarnings('ignore')


# In[2]:


get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')


# In[3]:


from nlp_func import *


# In[4]:


from modelling_pipeline import pipeline_nlp


# In[5]:


from plots_ import plot_history


# ## 2. Create some functions

# In[6]:


# Stopwords tokens
stopwords_en = set(stopwords.words('english'))
stopwords_en_json = pd.read_json("./stop_words_en.json")
new_stop = ['t', 'will']


# In[7]:


stopwords_punct = set(punctuation)
stoplist_combined = set.union(stopwords_punct, stopwords_en, stopwords_en_json, new_stop)

my_list = list(stoplist_combined)


# |WordNet tag   |Treebank tag   |
# |---|---|
# | n  | NN  |
# |  a | JJ  |
# |  s | JJ  |
# |  r |  RB |
# |  v |  VB |

# ## 3. Load data

# In[8]:


filepath_dict = {
    'yelp': 'data/yelp_labelled.txt',
    'amazon': 'data/amazon_cells_labelled.txt',
    'imdb': 'data/imdb_labelled.txt'
}

df_list = []
for source, filepath in filepath_dict.items():
    df = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t')
    df['source'] = source
    df_list.append(df)

df = pd.concat(df_list)
print(df.iloc[0])


# ## 3. Data analysis

# Below I'll have a look at the frist 5 columns of the data:

# In[9]:


df.head()


# In[10]:


df.shape


# The data has 2748 rows and 3 columns

# Now let's have a look at the distribution of the sentences in each source:

# In[11]:


df.source.value_counts().plot(kind='bar', grid=True, figsize=(10, 6))
plt.title("Distribution of Sentences by Source")
plt.ylabel('Count')


# In[12]:


df.source.value_counts(normalize=True).plot(kind='bar', grid=True, figsize=(10, 6))
plt.title("Distribution of Sentences by Source - Normalized values")
plt.ylabel('Count')


# **imdb** has less sentences than **yelp** and **amazon**

# In[13]:


df.sentence.map(len).hist(figsize=(10, 6))


# In[14]:


min(df.sentence.map(len))


# In[15]:


max(df.sentence.map(len))


# How balanced are the labels?

# In[16]:


df.label.value_counts().plot(kind='bar', grid=True, figsize=(10, 6))
plt.title("Distribution of labels")
plt.ylabel('Count')


# Sentence size range from 7 to 7944 characters.

# ### 3.1 NLP

# Now I will tokenize and lemmatize the sentences.
#     
# Tokenization separates the sentences into a set of tokens.
# 
# Lemmatization aims at reducing the inflectional forms of each word into a common base or root. Lemmatization, on the other hand, takes into consideration the morphological analysis of the words.
# 
# I still need to better understand about the stop words removal in the contaxt of Text Classification. I do believe (for now) that we should not remove the negative words at least. Some discussion [here](https://datascience.stackexchange.com/questions/31048/pros-cons-of-stop-word-removal), [here](https://www.researchgate.net/post/Does_Pre-processing_step_Remove_Stop_Word_effect_Sentiment_Analysis_result) and [here](https://stackoverflow.com/questions/40144473/do-we-need-to-use-stopwords-filtering-before-pos-tagging)

# In[17]:


tqdm.pandas()


# In[18]:


df['tokens'] = df['sentence'].progress_apply(lambda d: lemmatize_sent(d), df.sentence, [])
# df['tokens'] = df.sentence.progress_apply(lambda d: lemmatize_sent(d), [i for i in df.sentence.values], [])
# flatten = lambda data: reduce(lambda x, y: x + y, data, [])


# In[19]:


df['tokens'][:3]


# In[20]:


sum(df['sentence'].isnull())


# In[21]:


df.head()


# Checking the tokenization for the first 5 reviews (just to make sure it/how it worked):

# In[22]:


for text, tokens in zip(df['sentence'].head(5), df['tokens'].head(5)):
    print('Description: ', text[:60])
    print('Tokens: ', tokens[:45])
    print('-----------------')


# What are the most frequent tokens:

# In[23]:


alltokens = []
for token_list in df['tokens']:
    alltokens += token_list
counter = Counter(alltokens)


# In[24]:


counter.most_common(5)


# In[25]:


counter.most_common(10)


# Least frequent tokens:

# In[26]:


counter.most_common()[-10:-1]


# Finally, the weight of more common words which occur in all documents can be reduced. This is called *TFIDF* i.e Term Frequency times inverse document frequency. Now (as shown below) each of 2748 reviews is represented by 813 features, representing the tf-idf score for different ngrams.
# 
# min_df is set to 5, meaning that the word must be present in at least 5 documents to be kept
# 
# norm is set to l2

# In[27]:


tfidf = TfidfVectorizer(
    sublinear_tf=True,
    min_df=5,
    norm='l2',
    encoding='latin-1',
    ngram_range=(1, 4))
features = tfidf.fit_transform(
    list(df['tokens'].map(lambda tokens: ' '.join(tokens))))
labels = df['label']
features.shape


# In[28]:


tfidf_ = dict(zip(tfidf.get_feature_names(), tfidf.idf_))
tfidf_ = pd.DataFrame(columns=['tfidf']).from_dict(dict(tfidf_), orient = 'index')
tfidf_.columns = ['tfidf']


# In[29]:


tfidf_.sort_values(by = ['tfidf'], ascending = False)[:10]


# In[30]:


tfidf_.sort_values(by = ['tfidf'], ascending = True)[:10]


# Distribution of the TFIDF score (inludes up to 5 grams):

# In[31]:


tfidf_.tfidf.hist(ec = 'black')


# In[32]:


tfidf1 = TfidfVectorizer(
    sublinear_tf=True,
    min_df=5,
    norm='l2',
    encoding='latin-1',
    ngram_range=(1, 1))
features = tfidf1.fit_transform(
    list(df['tokens'].map(lambda tokens: ' '.join(tokens))))
labels = df['label']
features.shape


# In[33]:


tfidf_1 = dict(zip(tfidf1.get_feature_names(), tfidf1.idf_))
tfidf_1 = pd.DataFrame(columns=['tfidf']).from_dict(dict(tfidf_1), orient = 'index')
tfidf_1.columns = ['tfidf']


# Tokens with lowest TFIDF score:

# In[34]:


# tfidf_1.sort_values(by = ['tfidf'], ascending = True)[:10]


# In[35]:


plot_wc(tfidf_1.sort_values(by = ['tfidf'], ascending = True)[:20], 
        './wc_lowest_tfidf.png')


# Tokens with highest TFIDF score:

# In[36]:


plot_wc(tfidf_1.sort_values(by = ['tfidf'], ascending = False).head(20), 
        './wc_highest_tfidf.png')


# ## 4. Modelling

# In[37]:


X_train, X_test, y_train, y_test = train_test_split(
    df['tokens'],
    df['label'].values,
    random_state=24,
    shuffle=False,
    stratify=None,
    train_size=.8)
y_train = y_train.astype('int')
y_test = y_test.astype('int')


# In[38]:


tfidf = TfidfVectorizer(
    sublinear_tf=True,
    min_df=5,
    norm='l2',
    encoding='latin-1',
    ngram_range=(1, 4))
features = tfidf.fit_transform(
    list(df['tokens'].map(lambda tokens: ' '.join(tokens))))
labels = df['label']
features.shape


# ### 4.1 Baseline model 
# Model selection prior to using Keras

# In[39]:


# custom_vec = CountVectorizer(tokenizer=lemmatize_sent)
# cwm = custom_vec.fit_transform(df['sentence'])
# tokens = custom_vec.get_feature_names()
# # wm2df(cwm, tokens)


# In[40]:


results = pipeline_nlp(X_train, y_train, scoring='accuracy', n_folds=5)


# In[41]:


plt.figure(figsize=(15,8))
sns.boxplot(data=results)


# LinearSVC has an average accuracy of over 0.8

# How would the models perform for each source:

# In[42]:


dct = {}
for source in df['source'].unique():
    df_source = df[df['source'] == source]
    x = df_source['sentence'].values
    y = df_source['label'].values

    X_train_, X_test_, y_train_, y_test_ = train_test_split(
        x, y, train_size=.8, random_state=24)
    y_train = y_train.astype('int')
    y_test = y_test.astype('int')
    
    results = pipeline_nlp(X_train, y_train, scoring='accuracy', n_folds=5)
        
    plt.figure(figsize=(15,8))
    sns.boxplot(data=results).set_title(source)


# LinearSVC performed better than the other 4 models.

# ### 4.2 Parameter Tuning of Baseline Model

# In[43]:


run_gs = True
seed = 42

LinearSVC

pipeline = Pipeline(
    [('vect', CountVectorizer(analyzer=lemmatize_sent)),
     ('tfidf', TfidfTransformer()),
     ('clf',
      CalibratedClassifierCV(base_estimator=LinearSVC(random_state=seed)))])

print("done: pipeline")

if run_gs:

    # Set the grid parameters

    LinearSVC_grid = {
        'vect__ngram_range': [(1, 1), (1, 2),(1, 4)],
        'vect__max_features': (None, 500, 1000),
        'tfidf__use_idf': [True, False],
        'clf__base_estimator__loss': ['hinge', 'squared_hinge', 'perceptron'],
        'clf__base_estimator__penalty': ['l2', 'l1', None, 'elasticnet'],
        'clf__base_estimator__dual': [True, False]
    }

    print("StratifiedKFold")

    cross_validation = StratifiedKFold(n_splits=5)
    #     cross_validation = KFold(n_splits=5)

    print("done: classifier")

    grid_search = GridSearchCV(
        pipeline,
        param_grid=LinearSVC_grid,
        cv=cross_validation,
        scoring="recall_weighted",
        error_score=0)

    print('Will fit now')

    grid_search.fit(X_train, y_train)

    model = grid_search

    parameters = grid_search.best_params_

    print('Best score: {}'.format(grid_search.best_score_))

    print('Best parameters: {}'.format(grid_search.best_params_))

    print('Best estimator: {}'.format(grid_search.best_estimator_))

else:

    parameters = {
        'vect__ngram_range': [(1, 1)],
        'vect__max_features': [None],
        'tfidf__use_idf': [True],
        'clf__base_estimator__loss': ['hinge'],
        'clf__base_estimator__penalty': ['l2']
    }

    model = LinearSVC(**parameters)

    model.fit(X_train, y_train)


# In[44]:


# pipeline.get_params().keys()


# In[45]:


# save the model to disk
filename = 'LinearSVC_python3p6_CalibratedClassifierCV.dat'
pickle.dump(model, open(filename, 'wb'))


# ### 4.3 Evaluate predictions for Baseline Model

# In[46]:


y_pred = model.predict(X_test)


# In[47]:


# evaluate predictions


# accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# Classification report:
#label_list = sorted(list(y_test.unique()))
print (classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print ("Confusion Matrix \n", cm)

fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(cm, annot=True, fmt='d')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()


# In[48]:


pred_proba = model.predict_proba(X_test)
log_loss(y_test, pred_proba)


# In[49]:


pred_proba


# ### 4.4 Neural Network

# In[50]:


from keras.models import Sequential
from keras import layers


# In[51]:


from keras import backend as K
K.clear_session()


# In[52]:


X_train, X_test, y_train, y_test = train_test_split(
    df['tokens'],
    df['label'].values,
    random_state=24,
    shuffle=False,
    stratify=None,
    train_size=.8)
y_train = y_train.astype('int')
y_test = y_test.astype('int')


# In[53]:


tfidf = TfidfVectorizer(
    sublinear_tf=True,
    min_df=5,
    norm='l2',
    encoding='latin-1',
    ngram_range=(1, 4))
features = tfidf.fit_transform(
    list(X_train.map(lambda tokens: ' '.join(tokens))))
labels = df['label']
features.shape


# In[54]:


X_test  = tfidf.transform(list(X_test.map(lambda tokens: ' '.join(tokens))))


# In[55]:


X_test.shape


# In[56]:


X_train = features


# In[57]:


input_dim = X_train.shape[1]


# In[58]:


X_train.shape


# https://keras.io/getting-started/sequential-model-guide/
# 
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
# 

# In[59]:


model = Sequential()


# In[60]:


model.add(layers.Dense(10, input_dim = input_dim, activation='relu'))


# 618 (input_dim) parameters, with 10 neurons in the first hidden layer (output)

# In[61]:


model.add(layers.Dense(1, activation='sigmoid'))


# In[62]:


# Start training - configuraton of the lerning process:


# In[63]:


model.compile(
    loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[64]:


model.summary()


# *None* in the shape means it does not have a pre-defined number.

# For *dense_1 layer* the number os parameters is 6200. This value is obtained as: 

# In[65]:


(input_dim + 1) * 10 


# That means that for every hidden unit I have 618 (input_dim) input values + 1 bias times 10 neurons in the first layer

# For *dense_2* I have 10 input values * 1 neuron in the layer + 1 bias 

# In[66]:


10 + 1 * 1


# In[67]:


# training with the .fit() function


# It's necessary to specify the number of iterations. The completed iterations are called epochs. 
# 
# Batch size = how many sample in one forward/backward pass

# In[68]:


md = model.fit(
    X_train,
    y_train,
    epochs=100,
    verbose=False,
    validation_data=(X_test, y_test))


# In[69]:


md


# Use the .evaluate() method to measure the accuracy of the model.

# In[70]:


loss, accuracy = model.evaluate(X_train, y_train, verbose = False)


# In[71]:


print("Training accuracy: {:.4f}".format(accuracy))


# In[72]:


loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing accuracy:  {:.4f}".format(accuracy))


# The model is overfitting  - it nearly reached 100% accuracy for the training set. The testing accuracy is not higher than the baseline model.

# In[73]:


plot_history(md)


# It's clear that the model was training for too long.  As it's shown above, the loss of the validation rises again, which indicates overfitting. This happens at around 15 epochs.

# Also it's important to note that when training neural networks, one should use a separate testing and validation set. In this case, our testing and validation set are the same, since we have a smaller sample size. 

# In[74]:


#### Word Embeddings


# This represents dense word vectors. It collects more information with fewer dimensions.

# In[86]:


sent_train, sent_test, y_train2, y_tes2t = train_test_split(
#     df['sentence'].values,
    df['tokens'].values,
    df['label'].values,
    random_state=24,
    shuffle=False,
    stratify=None,
    train_size=.8)
y_train2 = y_train.astype('int')
y_test2 = y_test.astype('int')


# https://keras.io/preprocessing/text/:
# 
# num_words: the maximum number of words to keep, based on word frequency. Only the most common num_words-1 words will be kept.

# In[98]:


from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(sent_train)

X_train2 = tokenizer.texts_to_sequences(sent_train)
X_test2 = tokenizer.texts_to_sequences(sent_test)

vocab_size = len(tokenizer.word_index) + 1  # + 1 because of reserved 0 index. 
# Unknown words (words that are not in the vocabulary) are denoted in Keras with word_count + 1 since they can also hold some information

print(sent_train[2])
print(X_train2[2])


# max_len will pad sequences to same length. The padding will be done with zeros.

# In[99]:


from keras.preprocessing.sequence import pad_sequences

maxlen = 50
X_train2 = pad_sequences(X_train2, padding = 'post', maxlen=maxlen)
X_test2 = pad_sequences(X_test2, padding='post', maxlen=maxlen)


# In[100]:


print(X_train2[0, :])


# As shown above, the resulting vectors contains loads of zeros due to the padding. It's also clear that the first sentence is short.
