#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# ''' Natural Language Processing is wide field as we know before,Here we are discussing about basic applications of NLP That is Email classification into spam and ham which is popular evry one.
# we will take up an extremely popular use case of NLP - building a supervised machine learning model on text data.

# ## Problem Statement
# 
#  ''' Here we are going to automating to identify wheather the massage spam or ham. Nowdas spam messages are increasing randomly peoples are getting deffecult to identify wheather the message contains spam or not.So we are going to generate a classification model to classify wheather the message is ham or spam.
#  '''We will try to address this problem by building a text classification model which will automate the process.
#  The dataset we will use comes from Kaggle sample data set for LSTM for text classification Spam.csv and contains
#  5572 observations and 5 variables, as described below
#  
#    v1 : Indicating wheather message is spam or ham 
#    
#    v2 : messages
#    
#    v3 : Non
#    
#    v4 : Non 
#    
#    v5 : Non 
#    
#   

# ### Loading the required libraries and modules.

# In[4]:


# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder


get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[5]:


import abc 


# ### Loading the data and performing basic data checks

# In[6]:


df = pd.read_csv('spam.csv',delimiter=',',encoding='latin-1')
print(df.shape)
df.head()


# In[7]:


df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1,inplace=True)
df.info()


# In[8]:


print(df.shape)


# ""  Let us check the distribution of the target class which can be done using barplot.groups the 'V1' variables by counting     the number of their occurrences.
# It is evident that we have more occurrences of 'ham' than 'spam' in the target variable. Still, the good thing is that the difference is not significant and the data is relatively balanced.
# 
# 
# The baseline accuracy is important but often ignored in machine learning. It sets the benchmark in terms of minimum accuracy which the model should achieve.

# In[9]:


df.groupby('v1').v2.count().plot.bar(ylim=0)
plt.show()
print(2886/5572) #Baseline accuracy


# 0.5179468772433596 # Baseline accuracy

# ### Pre-processing the Raw Text and Getting It Ready for Machine Learning
# 
# Now, we are ready to build our text classifier. However, this is where things begin to get trickier in NLP. The data we have is in raw text which by itself, cannot be used as features. So, we will have to pre-process the text.
# 
# 
# For completing the above-mentioned steps, we will have to load the nltk package, 

# In[10]:


import nltk
nltk.download('stopwords')


# With nltk package loaded and ready to use, we will perform the pre-processing tasks. The first two lines of code below imports the stopwords and the PorterStemmer modules, respectively.
# 
# The third line imports the regular expressions library, ‘re’, which is a powerful python package for text parsing. To learn more about text parsing and the 're' library, please refer to the guide'Natural Language Processing – Text Parsing'(/guides/text-parsing).
# 
# The fourth to sixth lines of code does the text pre-processing discussed above.

# In[11]:


from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

stemmer = PorterStemmer()
words = stopwords.words("english")

df['processedtext'] = df['v2'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).
                                                         split() if i not in words]).upper())


# We will now look at the pre-processed data set that has a new column 'processedtext'.

# In[12]:


print(df.shape)
df.head()


# ## Upsampling for imbalnced classes

# In[13]:


df.groupby('v1').v2.count()


# In[14]:


from sklearn.utils import resample


# In[17]:


# Separate majority and minority classes
df_majority = df[df.v1=='ham']
df_minority = df[df.v1=='spam']
 
# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=4825,    # to match majority class
                                 random_state=123) # reproducible results
 
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
 
# Display new class counts
df_upsampled.v1.value_counts()


# In[ ]:





# In[ ]:





# ###  spliting train data sand test data 
# 
# We have already imported necessory pakages to split our data into train and test.
# 
# we are creating an array of the target variable, called 'target'.
# 
# 
# Then we are creating the training (X_train, y_train) and test set (X-test, y_test) arrays. It keeps 30% of the data for testing the model. The 'random_state' argument ensures that the results are reproducible.

# In[18]:


target = df_upsampled['v1']

X_train, X_test, y_train, y_test = train_test_split(df_upsampled['processedtext'], target, test_size=0.30, random_state=100)

print(df.shape); print(X_train.shape); print(X_test.shape)


# ### Converting Text to Word Frequency Vectors with TfidfVectorizer.
# 
# We have processed the text, but we need to convert it to word frequency vectors for building machine learning models. There are several ways to do this, such as using CountVectorizer and HashingVectorizer, but the TfidfVectorizer is the most popular one.
# 
# TF-IDF is an acronym that stands for 'Term Frequency-Inverse Document Frequency'. It is used as a weighting factor in text mining applications.
# 
# 
# Term Frequency (TF): This summarizes the normalized Term Frequency within a document.
# 
# 
# Inverse Document Frequency (IDF): This reduces the weight of terms that appear a lot across documents. In simple terms, TF-IDF attempts to highlight important words which are frequent in a document but not across documents. We will work on creating TF-IDF vectors for our documents.

# In[19]:


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer_tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)

train_tfIdf = vectorizer_tfidf.fit_transform(X_train.values.astype('U'))

test_tfIdf = vectorizer_tfidf.transform(X_test.values.astype('U'))

print(vectorizer_tfidf.get_feature_names()[:10])


# Let's look at the shape of the transformed TF-IDF train and test datasets. The following line of code performs this task.

# In[20]:


print(train_tfIdf.shape); print(test_tfIdf.shape)


# In[33]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 50)

classifier.fit(train_tfIdf, y_train)


# In[34]:


predRF = classifier.predict(test_tfIdf) 
print(predRF[:10])


# In[35]:


# Calculate the accuracy score
accuracy_RF = metrics.accuracy_score(y_test, predRF)
print(accuracy_RF)


# We see that the accuracy is increased to 97.5, Compairing to Naive Bayes Model nothing that much change any way both models give us a good score for this classification. 

# In[36]:



Conf_metrics_RF = metrics.confusion_matrix(y_test, predRF, labels=['ham', 'spam'])
print(Conf_metrics_RF)


# ### Conclusion
# 
#  
#  1 : Baseline Model Accuracy - 51.8%
#  
#  2 :   Accuracy achieved by Random Forest Classifier -99.4
#  
#  
# 

# In[37]:


import pickle


# In[38]:


pickle.dump(classifier, open('RF_classifier.pkl', 'wb'))


# In[ ]:




