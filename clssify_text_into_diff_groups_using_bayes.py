import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups

data = fetch_20newsgroups()
print data
print len(data.target_names)

categories = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc',
              'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
              'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles',
              'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt',
              'sci.electronics', 'sci.med', 'sci.space',
              'soc.religion.christian', 'talk.politics.guns',
              'talk.politics.mideast', 'talk.politics.misc',
              'talk.religion.misc']

# Trainin the data on these categories

train = fetch_20newsgroups(subset ='train',categories= categories)

# Testing the data for these categories

test = fetch_20newsgroups(subset ='test',categories= categories)

print train.data[5]  # will prints 5th article
print len(train.data)

# 11314   Articles train articals are there

from sklearn.feature_extraction.text import TfidfVectorizer
''' TfidfVectorizer :
It will redice the weights of few words in our article like :
millions,weapons,accidental,etc.'''

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# creating a model based on Multinomial Navie Bayes
model = make_pipeline(TfidfVectorizer(),MultinomialNB())

# Training the model with the train data
model.fit(train.data,train.target)

# Creating labels for the test data
labels =model.predict(test.data)
# Creating confusion matrix and heat map

from sklearn.metrics import confusion_matrix

mat = confusion_matrix(test.target,labels)
print mat

sns.heatmap(mat.T,square =True,annot=True,fmt='d',cbar=False
            ,xticklabels =train.target_names,
            yticklabels =train.target_names)

# plotting heatmap of confusion matrix

plt.xlabel('true label')
plt.ylabel('predicted label')
plt.savefig('confusion_matrix_after prediction')
plt.show()

# Predicting category on new data based on trained model

def predict_category(s,train=train,model=model):
    pred =model.predict([s])
    return train.target_names[pred[0]]


# Predicting each word classification

print predict_category('jesus')  #predicts : soc.religion.christian
print predict_category('president of india')  # talk.politics.misc
print predict_category('sending rocket to space station')  # sci.space
print predict_category('Benz is better than audi') # rec.autos 
