# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 11:19:27 2018

@author: Ethan
"""

'''
Citation Request:
  This dataset is public available for research. The details are described in [Cortez et al., 2009]. 
  Please include this citation if you plan to use this database:

  P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. 
  Modeling wine preferences by data mining from physicochemical properties.
  In Decision Support Systems, Elsevier, 47(4):547-553. ISSN: 0167-9236.

  Available at: [@Elsevier] http://dx.doi.org/10.1016/j.dss.2009.05.016
                [Pre-press (pdf)] http://www3.dsi.uminho.pt/pcortez/winequality09.pdf
                [bib] http://www3.dsi.uminho.pt/pcortez/dss09.bib

1. Title: Wine Quality 

2. Sources
   Created by: Paulo Cortez (Univ. Minho), Antonio Cerdeira, Fernando Almeida, Telmo Matos and Jose Reis (CVRVV) @ 2009
   
3. Past Usage:

  P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. 
  Modeling wine preferences by data mining from physicochemical properties.
  In Decision Support Systems, Elsevier, 47(4):547-553. ISSN: 0167-9236.

  In the above reference, two datasets were created, using red and white wine samples.
  The inputs include objective tests (e.g. PH values) and the output is based on sensory data
  (median of at least 3 evaluations made by wine experts). Each expert graded the wine quality 
  between 0 (very bad) and 10 (very excellent). Several data mining methods were applied to model
  these datasets under a regression approach. The support vector machine model achieved the
  best results. Several metrics were computed: MAD, confusion matrix for a fixed error tolerance (T),
  etc. Also, we plot the relative importances of the input variables (as measured by a sensitivity
  analysis procedure).
 
4. Relevant Information:

   The two datasets are related to red and white variants of the Portuguese "Vinho Verde" wine.
   For more details, consult: http://www.vinhoverde.pt/en/ or the reference [Cortez et al., 2009].
   Due to privacy and logistic issues, only physicochemical (inputs) and sensory (the output) variables 
   are available (e.g. there is no data about grape types, wine brand, wine selling price, etc.).

   These datasets can be viewed as classification or regression tasks.
   The classes are ordered and not balanced (e.g. there are munch more normal wines than
   excellent or poor ones). Outlier detection algorithms could be used to detect the few excellent
   or poor wines. Also, we are not sure if all input variables are relevant. So
   it could be interesting to test feature selection methods. 

5. Number of Instances: red wine - 1599; white wine - 4898. 

6. Number of Attributes: 11 + output attribute
  
   Note: several of the attributes may be correlated, thus it makes sense to apply some sort of
   feature selection.

7. Attribute information:

   For more information, read [Cortez et al., 2009].

   Input variables (based on physicochemical tests):
   1 - fixed acidity
   2 - volatile acidity
   3 - citric acid
   4 - residual sugar
   5 - chlorides
   6 - free sulfur dioxide
   7 - total sulfur dioxide
   8 - density
   9 - pH
   10 - sulphates
   11 - alcohol
   Output variable (based on sensory data): 
   12 - quality (score between 0 and 10)
   13 - color (0 = red, 1 = white)


8. Missing Attribute Values: None
'''


import csv
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm

def load_data(filename_red='winequality-red_formatted.csv', filename_white='winequality-white_formatted.csv'):
    '''
    loads attributes from 2 csv data sets and returns single list
    1 data set each for red and white vino verde Portuguese wines
    skips first line of white wine which lists classifiers of attributes as red wine file contains this
    ::param filename_red: data set containing attributes of red wines; indicated by 13th attribute "0"
    ::param filename_white: data set containing attributes of white wines; indicated by 13th attribute "1"
    '''
    reader = csv.reader(open(filename_red, 'r', encoding='utf-8', errors='ignore'))
    data = []
    for line in reader:
        data.append(line)
    reader = csv.reader(open(filename_white, 'r', encoding='utf-8', errors='ignore'))
    for line in reader:
        data.append(line)
    return data

wine_data = load_data()

def string_to_flt(data):
    float_list = []
    float_list.append(data[0])
    for wine in data[1:]:
        float_wine = []
        for attr in wine:
            float_wine.append(float(attr))
        float_list.append(float_wine)
    return float_list
            
wine_data = string_to_flt(wine_data)
'''
def train_vectorizer(tweets):
    cv = CountVectorizer(strip_accents='unicode', min_df=3)
    cv.fit(tweets)
    return cv

def vectorize_tweets(tweet_vectorizer, tweets):
    return tweet_vectorizer.transform(tweets)

def vectorize_sentiments(data):
    return np.array([int(row[0]) for row in data], dtype=np.int)

# Load raw data into nested lists
data = load_data()

# This code extracts only the tweets from the loaded data into a list of strings
tweets = [row[5] for row in data]

# Create a vectorizer - this will allow us to map each tweet to a vector of word counts
tweet_vectorizer = train_vectorizer(tweets)

# Test your tweet vectorizer using an example - note that the vectorize_tweets function
# takes a LIST OF TWEETS (strings), as a parameter and not a single tweet.
example_tweet = 'Apples, bananas, and kiwis are my favourite fruits to bring to the zoo!'
example_tweet_vector = vectorize_tweets(tweet_vectorizer, [example_tweet])

# Now apply the tweet vectorizer to all tweets - pay attention to the type and shape of what this function returns
tweet_vectors = vectorize_tweets(tweet_vectorizer, tweets)

# Get the sentiment labels from the data and organize it into a vector
sentiment_vector = vectorize_sentiments(data)

### ADVICE: You should now take a look at tweet_vectors and sentiment_vector. The idea here is that, for 
###         each tweet vector, you have a corresponding sentiment label in the sentiment_vector. If you 
###         are using the full dataset, you should check that the shape of tweet_vectors is (1600000, X)
###         and that the shape of sentiment_vector is (1600000,) or (1600000,1). Here X can be any number
###         and is the number of features in each vector. In other words, X will be the size of the vocabulary
###         extracted from the data. If you use a smaller subset of the data, you should expect X to be 
###         smaller, too.

### From here, you'll now want to implement your learn functions as well as your visualizations. Play with the raw
### data, the vectorized data, or your results from applying machine learning to get ideas for visualization. 
### Remember, the raw data is loaded simply as text data - so there are many functions you can use to look for
### tweets that contain specific words or terms you're interested in analyzing. Have fun and good luck!
'''