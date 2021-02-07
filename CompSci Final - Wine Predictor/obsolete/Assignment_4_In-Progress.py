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
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler as minMaxScaler
from sklearn.neighbors import KNeighborsClassifier


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

test_wines = create_test_data() # create training list


'''
START OF MACHINE LEARNING MODEL 1
this model will take the first 11 attributes as the vector, and the quality (attr 12)
    as the value which the model will predict
So:
    X = (P x N) where P is number of wines and N = 11 (vector is 11 attributes)
    Y = (P) where P is number of wines populated by quality scores
    After the model runs, it will be able to take a new vector and predict quality
'''


scaler = minMaxScaler(feature_range=(-1,1)) #create min_max_scaler, range -1 - 1

def feature_vectorizer(wine_data):
    '''
    creates "X" to be used in SKL
    1 vector for each wine, stored in array
    matrix (P x N) = (6497 x 11)
    skips data[0] as it contains classifiers for each attribute
    scales features to between -1 : 1 to avoid any features taking precedence due to magnitude
    returns matrix of wine_vectors
    ::param wine_data: nested list of 13 attributes for each wine within
    '''
    X = np.array([[wine_data[0][0:11]]])
    for wine in wine_data[1:]:
        X = np.append(X, wine[0:11])
    X = X.reshape((len(wine_data)), 11)
    scaler.fit(X)
    X = scaler.transform(X)
    return X

X = feature_vectorizer(training_wines)    # create matrix X to be used in machine learning 1
test_X = feature_vectorizer(test_wines) #vectorize test wines    


    



        
        

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


y1 = quality_vectorizer(training_wines)  # creates matrix y to be used in machine learning 1
test_y1 = quality_vectorizer(test_wines)  #creates matrix test_y1 to grade predictions of test set


my_model_1 = KNeighborsClassifier()
my_model_1.fit(X, y1)

'''
above code trains model according to sklearn fit function
earlier I tried linearSVC, but the prediction rating of KNN is better
'''

#sanity_quality = my_model_1.predict(X) #use model to predict quality training data
#test_quality = my_model_1.predict(test_X) #use model to predict quality test wine qualities



def test_model_1(number_reds = 599, number_whites = 3898, buffer = False, data = test_wines):
    '''
    Returns nested list of True/False statements showing if model predicts quality of wines accurately
    Parameters set for rating the predictions of the test set, but can be reset for the training set
    First list contains prediction data for reds
    Second list contains prediction data for whites
    prediction for wine quality must be accurate within buffer (0 or 1) of true quality for True
    ::param number_reds: number red wines to test, limited to 599, autamaticcaly includes all test reds
    ::param number_whites: number white wines to test, limited to 3898, automatically includes all test whites
    ::param buffer: when True adds room for error in predictions of +/-1
    '''
    wines = []
    reds = []
    whites = []
    reds_vector = feature_vectorizer(data[:number_reds])
    reds_quality = my_model_1.predict(reds_vector)
    for i in range(len(reds_quality)):
        reds.append(reds_quality[i] == data[i][11] or reds_quality[i] == (data[i][11] + buffer) or reds_quality[i] == (data[i][11] - buffer))
    wines.append(reds)
    whites_vector = feature_vectorizer(data[number_reds:(number_reds + number_whites)])
    whites_quality = my_model_1.predict(whites_vector)
    for i in range(len(whites_quality)):
        whites.append(whites_quality[i] == data[i+number_reds][11] or whites_quality[i] == (data[i+number_reds][11] + buffer) or whites_quality[i] == (data[i+number_reds][11] - buffer))
    wines.append(whites)
    return wines
 
prediction_results_sanity_quality = test_model_1(number_reds=1000, number_whites=1000, buffer=0, data=training_wines) #creates nested list of true/falses for prediction of training list
prediction_results_quality = test_model_1(number_reds = 599, number_whites = 3989, buffer = 0, data = test_wines) #creates nested list of true/falses for prediction of test list




def model_accuracy(results):
    '''
    turns prediction results into % score for reds and whites
    ::param results: nested list of 2 lists with true/false statements, first is reds, second is whites
    '''
    accuracy = []
    count_red = 0
    count_white = 0
    count = 0
    if type(results[0]) == list:
            for i in results[0]:
                if i == True:
                    count_red += 1
            accuracy.append(round(count_red / len(results[0]), 2))
            for i in results[1]:
                if i == True:
                    count_white += 1
            accuracy.append(round(count_white / len(results[1]), 2))
            return '{}% reds predicted correctly, {}% whites predicted correctly'.format(accuracy[0], accuracy[1])
    else:
        for i in results:
            if i == True:
                count += 1
        accuracy = (round(count / len(results)))
    return '{}% predicted correctly'.format(accuracy)
    
score_quality_test = model_accuracy(prediction_results_quality) #prediction score of model on test data
score_quality_sanity = model_accuracy(prediction_results_sanity_quality) #prediction score of model on training data

            



'''
Start of model 2
This model uses the same 11 features as model 1 which were used to predict quality
and instead uses the features to predict color of wine
this meanst the 'y' that fit will use is the '13th' feature of the data set instead of the 12th
this 13th feature is a 0 or 1
0 = red
1 = white
'''

def color_vectorizer(wine_data):
    '''
    return vectorized (len(wine_data) x 1) array of wines
    ::param wine_data: a list of wines to vectorize
    '''
    y = np.array([wine_data[0][12]])
    for wine in wine_data[1:]:
        y = np.append(y, wine[12])
    return y


#vectorize wine sets
y2 = color_vectorizer(training_wines)
test_y2 = color_vectorizer(test_wines)

#fit to linearSVC, while it didn't work well for quality it works quite well for color
my_model_2 = LinearSVC()
my_model_2.fit(X, y2)


def test_model_2(number_reds = 599, number_whites = 3898, data = test_wines, ):
    '''
    Returns nested list of True/False statements showing if model predicts color of wines accurately
    First list contains prediction data for reds
    Second list contains prediction data for whites
    set to test predictions of test set if no paramaters set
    ::param number_reds: number red wines to test, limited to 599, automatically includes all test reds
    ::param number_whites: number white wines to test, limited to 3898, automatically includes all test whites
    ::param data: list of wines to grade predictions against
    '''
    test_wines = []
    test_reds = []
    test_whites = []
    test_reds_vector = feature_vectorizer(data[:number_reds])
    test_reds_color = my_model_2.predict(test_reds_vector)
    for i in range(len(test_reds_color)):
        test_reds.append(test_reds_color[i] == data[i][12])
    test_wines.append(test_reds)
    test_whites_vector = feature_vectorizer(data[number_reds:(number_reds + number_whites)])
    test_whites_color = my_model_2.predict(test_whites_vector)
    for i in range(len(test_whites_color)):
        test_whites.append(test_whites_color[i] == data[i+number_reds][12])
    test_wines.append(test_whites)
    return test_wines


#prediction true/false lists
prediction_results_sanity_color = test_model_2(number_reds = 1000, number_whites = 1000, data = training_wines)
prediction_results_color_test = test_model_2()

#prediction scores    
score_color_test = model_accuracy(prediction_results_color_test) #test set
score_color_sanity = model_accuracy(prediction_results_sanity_color) #training set


#get rid of unneeded variables
del prediction_results_quality
del prediction_results_sanity_quality 
del prediction_results_sanity_color
del prediction_results_color_test
del wine_data_float
del wine_data_string
del training_wines
del test_wines
del test_y1
del test_y2
del test_X


'''
STARTING PLOT PORTION HERE
'''

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import matplotlib.lines as mlines


#set TSNE model up to condense passed factors into two which will explain most variance in 2000 data points
# as possible
n_sne = 2000
tsne = TSNE(n_components=2, verbose = 1, perplexity = 40, n_iter=300) 


'''
Code following is first visualization
First fits tsne model to training data set with subjective ratings out of 10 as labels
then creates a plot with the two TSNE factors as x and y axes with legend matching
    data points to their ratings
The resulting scatter plot is, frankly, a mess.  It seems as though there is no rhyme or reason
    to the way wines are plotted.  This makes sense, as the predictive score of the model is 
    only around .40 for the original training data.  However, as these are subjective ratings,
    it does makes sense.  It suggests that while the ratings are clearly subjective and reflect
    differences in taste, and that what people are tasting for is most likely personal and inconsistent
    even at the individual level.  However, that the model predicts well above chance shows that there 
    is indeed something in the wine that people are able to clue in on and call quality or not quality.
    It's also distinctly possible that a better learning model could be created which have a higher 
    prediction score than mine does, especially as the ratings were actually kind of close to
    predicting something complex given the simplicity of the model doing it.
'''    
tsne_results_1 = tsne.fit_transform(X, y1)

fig, ax = plt.subplots(figsize=(10, 5))
plt.scatter(tsne_results_1[:,0], tsne_results_1[:,1], c=y1, cmap = 'tab10')
plt.xlabel('TSNE Factor 0')
plt.ylabel('TSNE Factor 1')
plt.title('Wine Quality')
line0 = mlines.Line2D([], [], color = 'tab:blue', label = '1')
line1 = mlines.Line2D([], [], color = 'tab:orange', label = '2')
line2 = mlines.Line2D([], [], color = 'tab:green', label = '3')
line3 = mlines.Line2D([], [], color = 'tab:red', label = '4')
line4 = mlines.Line2D([], [], color = 'tab:purple', label = '5')
line5 = mlines.Line2D([], [], color = 'tab:brown', label = '6')
line6 = mlines.Line2D([], [], color = 'tab:pink', label = '7')
line7 = mlines.Line2D([], [], color = 'tab:grey', label = '8')
line8 = mlines.Line2D([], [], color = 'tab:olive', label = '9')
line9 = mlines.Line2D([], [], color = 'tab:cyan', label = '10')
    
ax.legend(handles=[line0, line1, line2, line3, line4, line5, line6, line7, line8, line9], title = 'score out of 10', loc = 'best')
plt.show

plt.figure()


'''
next portion of code plots data from second model
first performs dimensionality reduction fit on training data with y2 which has color labels
Then begins a plot using the TSNE factors it calculated as axes for each of the 2000 wines in the data set
As seen in the plot, the distinction of wine color is quite clear.  This shows that there are clear differences
    at the chemical level between red and white wines the model is picking up on, allowing it to predict what 
    color a wine of this varietal is at very high accuracy
'''

tsne_results_2 = tsne.fit_transform(X, y2) #fit TSNE to data with color labels


fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(tsne_results_2[:,0], tsne_results_2[:,1], c=y2, cmap = 'viridis')
plt.xlabel('TSNE Factor 0')
plt.ylabel('TSNE Factor 1')
plt.title('Wine Color')
purple_patch = mlines.Line2D([], [], color='purple', label='red wines')
yellow_patch = mlines.Line2D([], [], color='yellow', label='white wines')
ax.legend(handles=[purple_patch, yellow_patch], loc = 'best')
plt.show()  



##next two lines print accuracy statements of models AFTER plots, for cleanliness
print('model 1 accuracy on test set: {}, \nmodel 1 accuracy on training set: {}\n'.format(score_quality_test, score_quality_sanity))


print('model 2 accuracy on test set: {} \nmodel 2 accuracy on training set: {}\n'.format(score_color_test, score_color_sanity))


'''
START SORT PORTION OF ASSIGNMENT
'''

'''
I chose heapsort, comments describe it's methods quite well

I tested the efficiency against normal selection because wikipedia describes heapSort as a more efficient
    version of selection sort sort.  I used a randomly generated list of 1,000,000 
    elements.  Selection sort made 4,999,950,000 comparisons.  This is within range of it's 
    theoretical maximum of 5,000,000,000.  This version of heapsort had only 3,019,495 comparisons
    to sort the exact same list.  This only 3x the length of the list itself, and 1655.89 fewer
    comparisons than selection sort made.  I didn't build in a timer, but I can say that heapSort
    was pretty fast; seconds to sort a list that size.  Selection sort was minutes.  I 
    actually left the program running to work on other assignments it took so long, checking
    back every 30 seconds to a minute.  
    I included a call to heapSort a randomly generated list of 10,000, as well as a copy of the selection
    sort program we wrote in class (without a call), just to cover my bases.
    
But overall I'd say this is heads and shoulders ahead of selection sort, overall a pretty good sorting 
    algorithm compared what I've seen so far.
    
    '''
    
import random
my_randoms = random.sample(range(10001), 10000)
copy = []
for i in my_randoms:
    copy.append(i)

def heapSort(heap):
    '''
    Assumes a broken binary heap
    Compares last parent node to children, and sifts any larger children up heap
    Once largest node at root, switches for last index of list
    removes last index from binary heap, and starts again from (new) last parent node
    repeats until root node sorted
    returns statement with number of comparisons, recursions, and swaps performed by keeping variables global
    ::param heap: a list to be sorted
    '''
    global comparisons
    global recursions
    global swaps
    comparisons = 0
    swaps = 0
    recursions = 0
    global heapSize
    heapSize = len(heap)
    
    def Switch(i, x):
        '''
        Switches the numbers contained at given indices
        ::param i: index of first number to be switched
        ::param x: index of second number to be switched
        '''
        temp = heap[i]
        heap[i] = heap[x]
        heap[x] = temp
    
    
    def maxHeapify(heap, i):
        '''
        Sets node as "larger"
        Checks node has children in heap
        If node has children, compares it to them
        If left node is larger, sets it as such
        If right node is larger than "larger," sets it as such
        If "larger" is not the index of original node, switches the values of largest child and parent
             If switch is made, calls itself recursively to compare original parent to new children
        Each comparison made increases "comparison" count
        Each switch made increases "swap" count
        Each call to function increases "recursion" count
        ::param heap: list to be sorted
        ::param i: node to compare to children
        '''
        global comparisons
        global swaps
        global recursions
        global heapSize
        recursions += 1
        largest = i
        l = i * 2 + 1
        r = i * 2 + 2
        if l <= (heapSize - 1): #if there is a left child add a comparison to count and compare to parent
            comparisons += 1          
            if heap[l] > heap[i]:
                largest = l
            if r <= (heapSize - 1): #if there is a right child add a comparison to count and compare to largest
                comparisons += 1
                if heap[r] > heap[largest]:
                    largest = r
            if largest != i: #if the largest is not the parent, switch parent with larger child
                swaps += 1
                Switch(i, largest)
                maxHeapify(heap, largest)
        return comparisons, recursions            
                 
    
    def buildMaxHeap(heap):
        '''
        Repairs broken heap
        starts at last parent node and moves to root
        sifts largest children up by calling MaxHeapify
        ::param heap: list to be sorted
        '''
        for i in range(heapSize // 2 + 1, -1, -1):
            maxHeapify(heap, i)
        
    
    buildMaxHeap(heap)                  #repair broken heap and build maxheap
    for i in range(heapSize - 1, 0, -1):#for loop counting from last index in heap and moving to 0 by 1
        Switch(0, heapSize - 1)             #Switches largest element from root node with element in last node
        heapSize = heapSize - 1             #Removes last node, containing largest element from heap
        maxHeapify(heap, 0)                 #Call maxheapify on remaining heap to re-sort, starts from root and moves down to reduce number of possible recursions that comes from starting at bottom node each call
    return (print('heapSort counts:: comparisons = {}, recursions = {}, swaps = {}'.format(comparisons, recursions, swaps)))  #Once list is sorted, returns statement of counts


heapSort(my_randoms)

def selection_sort(inlist):
    global comparisons
    comparisons = 0 
    for i in range(len(inlist)):
       # Find the smallest remaining element
       min_index = i
       min_val = inlist[i]
       for j in range(i+1,len(inlist)):
          comparisons += 1
          if inlist[j] < min_val:
            min_val = inlist[j]
            min_index = j

       # Swap it to the left side of the list
       inlist[min_index] = inlist[i]
       inlist[i] = min_val
    return(print('comparisons: {}'.format(comparisons)))
    #return inlist