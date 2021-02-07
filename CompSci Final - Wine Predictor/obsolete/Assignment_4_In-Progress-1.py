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
###Look into TSNE for data visualization rather than standardizing and using PCA

import matplotlib.pyplot as plt
from matplotlib.mlab import PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import csv
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.svm import LinearSVC
#%matplotlib inline

def load_data(filename_red='winequality-red_formatted.csv', filename_white='winequality-white_formatted.csv'):
    '''
    loads attributes from 2 csv data sets
    1 data set each for red and white vino verde Portuguese wines
    first line of red wine file has classifiers of attributes
    returns nested list with attributes of 1599 red and 4898 white wines as type(str)
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

wine_data_string = load_data() #load csv data into Python

def string_to_flt(data = wine_data_string):
    '''
    returns a new nested list populated with wine scores of type(float) instead of type(string)
    ::param data: the combined data set of red/white wine attributes
    '''
    float_list = []
    float_list.append(data[0])
    for wine in data[1:]:
        float_wine = []
        for attr in wine:
            float_wine.append(float(attr))
        float_list.append(float_wine)
    return float_list

        
wine_data_float = string_to_flt(wine_data_string) #converts wine_data attributes to floats

def create_training_data(data = wine_data_float):
    '''
    Returns new list of wine data with equal number red and white wines for model
    to train with
    Feature data for 1000 wines of each color included
    Removes include first nested list containing category names for features
    ::param data: the combined data set of red/white wine; features in type(float)
    '''
    training_data = []
    for wine in data[1:1001]:
        training_data.append(wine)
    for wine in data[1600:2600]:
        training_data.append(wine)
    return training_data

training_wines = create_training_data() #create training list


def create_test_data(data = wine_data_float):
    '''
    Returns new list of wine data with remaining wines not used for training
    This new list will be used to test model on
    Feature data for 599 red wines and 3898 white wines included
    Removes first nested list containing category names for features
    ::param data: the combined data set of red/white wine; features type(float)
    '''
    test_data = []
    for wine in data[1001:1600]:
        test_data.append(wine)
    for wine in data[2600:]:
        test_data.append(wine)
    return test_data

test_wines = create_test_data() # create test list



'''
START OF MACHINE LEARNING MODEL 1
this model will take the first 11 attributes as the vector, and the quality (attr 12)
    as the value which the model will predict
So:
    X = (P x N) where P is number of wines and N = 11 (vector is 11 attributes)
    Y = (P) where P is number of wines populated by quality scores
    After the model runs, it will be able to take a new vector and predict quality
'''



def feature_vectorizer(wine_data):
    '''
    creates "X" to be used in SKL
    1 vector for each wine, stored in array
    matrix (P x N) = (6497 x 11)
    skips data[0] as it contains classifiers for each attribute
    returns matrix of wine_vectors
    ::param wine_data: nested list of 13 attributes for each wine within
    '''
    X = np.array([[wine_data[0][0:11]]])
    for wine in wine_data[1:]:
        X = np.append(X, wine[0:11])
    X = X.reshape((len(wine_data)), 11)
    return X
        
X = feature_vectorizer(training_wines)    # create matrix X to be used in machine learning 1
    
min_max_scaler = MinMaxScaler()
scaled_X = min_max_scaler.fit_transform(X)
    
    
def quality_vectorizer(wine_data):
    '''
    creates "y" to be used by SKL
    1 vector for each wine, stored in array
    matrix (P) = (6497 x 11)
    skips data[0] as it contains classifiers for each attribute
    returns matrix of quality vectors
    ::param wine_data: nested list of 13 attributes for each wine within
    '''
    y = np.array([wine_data[0][11]])
    for wine in wine_data[1:]:
        y = np.append(y, wine[11])
    return y


y_1 = quality_vectorizer(training_wines)  # creates matrix y to be used in machine learning 1


my_model_1 = LinearSVC()
my_model_1.fit(scaled_X, y_1)
'''
above code trains model according to sklearn fit function
'''

test_X = feature_vectorizer(test_wines) #vectorize test wines
scaled_test_X = min_max_scaler.transform(test_X)
test_quality = my_model_1.predict(scaled_test_X) #use model to predict test wine qualities

def test_model_1(number_reds = 599, number_whites = 3898, buffer = 0, data = test_wines):
    '''
    Returns nested list of True/False statements showing if model predicts quality of test wines accurately
    First list contains prediction data for reds
    Second list contains prediction data for whites
    prediction for wine quality must be accurate within buffer (0 or 1) of true quality for True
    ::param number_reds: number red wines to test, limited to 599, autamaticcaly includes all test reds
    ::param number_whites: number white wines to test, limited to 3898, automatically includes all test whites
    ::param buffer: set a buffer zone to which quality prediction must be accurate to, automatically at 0
    '''
    test_wines = []
    test_reds = []
    test_whites = []
    test_reds_vector = feature_vectorizer(data[:number_reds])
    test_reds_quality = my_model_1.predict(test_reds_vector)
    for i in range(len(test_reds_quality)):
        test_reds.append(test_reds_quality[i] == data[i][11] or test_reds_quality[i] == (data[i][11] + buffer) or test_reds_quality[i] == (data[i][11] - buffer))
    test_wines.append(test_reds)
    test_whites_vector = feature_vectorizer(data[599:(599 + number_whites)])
    test_whites_quality = my_model_1.predict(test_whites_vector)
    for i in range(len(test_whites_quality)):
        test_whites.append(test_whites_quality[i] == data[i+599][11] or test_whites_quality[i] == (data[i+599][11] + buffer) or test_whites_quality[i] == (data[i+599][11] - buffer))
    test_wines.append(test_whites)
    return test_wines
        
prediction_results = test_model_1(599, 3989, 0)

def model_1_accuracy(results = prediction_results):
    if len(results) > 1:
        accuracy = []
        count_red = 0
        count_white = 0
        for i in results[0]:
            if i == True:
                count_red += 1
        accuracy.append(round(count_red / len(results[0]), 2))
        for i in results[1]:
            if i == True:
                count_white += 1
        accuracy.append(round(count_white / len(results[1]), 2))
        return accuracy
    
ratio_correct = model_1_accuracy()
            

### As of right now don't know how to make "real buffer," so it has to stay at either 0 or 1



'''
Start of model 2
This model uses the same 11 features as model 1 which were used to predict quality
and instead uses the features to predict color
this meanst the 'y' that fit will use is the '13th' feature of the data set instead of the 12th
this 13th is a 0 or 1
0 = red
1 = white
'''

def color_vectorizer(wine_data):
    y = np.array(wine_data[0][12])
    for wine in wine_data[1:]:
        y = np.append(y, wine[12])
    return y

y_2 = color_vectorizer(training_wines)

my_model_2 = LinearSVC()
my_model_2.fit(X, y_2)
       

'''
Alternative option: unsupervised clustering into 2 clusters
'''

from sklearn import cluster
from mpl_toolkits.mplot3d import Axes3D


from sklearn.cluster import KMeans

k_means = KMeans(n_clusters=10)
kmeans = k_means.fit(X)


print(k_means.cluster_centers_)

'''
def train_vectorizer(wines):
    cv = CountVectorizer(strip_accents='unicode', min_df=3)
    cv.fit(wines)
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
tweet_vectorizer = train_vectorizer(wines)

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