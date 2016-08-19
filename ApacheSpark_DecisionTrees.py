
# coding: utf-8

# # Spark-Mllib: Decision trees 

# Decision Trees (DTs) and their ensembles are popular methods for the machine learning task of a non-parametric supervised learning method used for classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.
# 
# 
# Some advantages of decision trees are:
# <li>Simple to understand and to interpret. Trees can be visualised.</li>
# <li>Requires little data and feature preparation. </li>
# <li>Able to handle both numerical and categorical data. </li>
# <li>Able to handle multi-output problems.</li>
# <li>Possible to validate a model using statistical tests. </li>
# <li>Performs well even if its assumptions are somewhat violated by the true model from which the data were generated.</li>

# <h2>Decision Trees: Algorithmic Background </h2>
# <p>
# <img src="https://databricks.com/wp-content/uploads/2014/09/decision-tree-example.png" alt="Mountain View" style="width:304px;height:228px;">
# 
# 
# <p>
# A model is learned from a training dataset by building a tree top-down. The if-else statements, also known as splitting criteria, are chosen to maximize a notion of information gain — it reduces the variability of the labels in the underlying (two) child nodes compared the parent node. The learned decision tree model can later be used to predict the labels for new instances.
# 
# These models are interpretable, and they often work well in practice. Trees may also be combined to build even more powerful models, using ensemble tree algorithms. Ensembles of trees such as random forests and boosted trees are often top performers in industry for both classification and regression tasks.
# 
# 

# ## Getting the data and creating the RDD

# <h2>Initializing Spark</h2>
# <p>The first thing a Spark program must do is to create a SparkContext object, which tells Spark how to access a cluster.
# To create a SparkContext you first need to build a SparkConf object that contains information about your application.

# In[1]:


import findspark
findspark.init()
from pyspark import SparkConf, SparkContext,SQLContext
sc = SparkContext("local","App:DecisionTrees")
sqlContext = SQLContext(sc)


# <h3>Training Dataset </h3>
# <p> Get the training data from web. You have data in the Data/KDD folder 

# In[2]:

import urllib
#f = urllib.urlretrieve ("http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz", "Data/kddcup.data.gz")
data_file = "Data/KDD/kddcup.data.gz"
raw_data = sc.textFile(data_file)
print "Train data size is {}".format(raw_data.count())


# <h3>Testing Dataset </h3>
# <p> Get the testing data from web. You have data in the Data/KDD folder 

# In[3]:

#ft = urllib.urlretrieve("http://kdd.ics.uci.edu/databases/kddcup99/corrected.gz", "Data/KDD/corrected.gz")
test_data_file = "Data/KDD/corrected.gz"
test_raw_data = sc.textFile(test_data_file)
print "Test data size is {}".format(test_raw_data.count())


# ## Problem Statement 
# <p>The task for the classifier learning was to learn a predictive model (i.e. a classifier) capable of distinguishing between legitimate and illegitimate connections in a computer network.

# In[4]:

from pyspark.mllib.regression import LabeledPoint
from numpy import array

csv_data = raw_data.map(lambda x: x.split(","))
test_csv_data = test_raw_data.map(lambda x: x.split(","))

protocols = csv_data.map(lambda x: x[1]).distinct().collect()
services = csv_data.map(lambda x: x[2]).distinct().collect()
flags = csv_data.map(lambda x: x[3]).distinct().collect()


# Python lists to create_labeled_point function.

# In[5]:

def create_labeled_point(line_split):
    # leave_out = [41]
    clean_line_split = line_split[0:41]
    # convert protocol to numeric categorical variable
    try: 
        clean_line_split[1] = protocols.index(clean_line_split[1])
    except:
        clean_line_split[1] = len(protocols)
    # convert service to numeric categorical variable
    try:
        clean_line_split[2] = services.index(clean_line_split[2])
    except:
        clean_line_split[2] = len(services)
    # convert flag to numeric categorical variable
    try:
        clean_line_split[3] = flags.index(clean_line_split[3])
    except:
        clean_line_split[3] = len(flags)
    # convert label to binary label
    attack = 1.0
    if line_split[41]=='normal.':
        attack = 0.0     
    return LabeledPoint(attack, array([float(x) for x in clean_line_split]))

training_data = csv_data.map(create_labeled_point)
test_data = test_csv_data.map(create_labeled_point)
print ("Done!")


# <h2>Training a classifier</h2>
# <p>We are now ready to train our classification tree.
# We will keep the maxDepth value small. 
# This will lead to smaller accuracy, but we will obtain less splits so later on we can better interpret the tree.
# In a production system you can try to increase this value in order to find a better accuracy.

# In[6]:

from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from time import time
# Build the model
t0 = time()
tree_model = DecisionTree.trainClassifier(training_data, numClasses=2, 
                                          categoricalFeaturesInfo={1: len(protocols), 2: len(services), 3: len(flags)},
                                          impurity='gini', maxDepth=4, maxBins=100)
tt = time() - t0
print "Classifier trained in {} seconds".format(round(tt,3))


# ## Evaluating the model

# In order to measure the classification error on our test data, we use map on the test_data RDD and the model to predict each test point class.

# In[7]:

predictions = tree_model.predict(test_data.map(lambda p: p.features))
labels_and_preds = test_data.map(lambda p: p.label).zip(predictions)
t0 = time()
test_accuracy = labels_and_preds.filter(lambda (v, p): v == p).count() / float(test_data.count())
tt = time() - t0
print "Prediction made in {} seconds. Test accuracy is {}".format(round(tt,3), round(test_accuracy,4))


# <h2>Interpreting the model</h2>
# Understanding our tree splits is a great exercise in order to explain our classification labels in terms of predictors and the values they take. Using the toDebugString method in our three model we can obtain a lot of information regarding splits, nodes, etc.

# In[8]:

print "Learned classification tree model:"
print tree_model.toDebugString()

