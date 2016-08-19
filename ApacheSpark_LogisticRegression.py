
# coding: utf-8

# # Spark-Mllib: Logistic Regression

# Logistic regression, despite its name, is a linear model for classification rather than regression. 
# Logistic regression is also known in the literature as logit regression, maximum-entropy classification (MaxEnt) or
# the log-linear classifier. 
# In this model, the probabilities describing the possible outcomes of a single trial are modeled using a logistic function.

# <h2><b>Apache Spark </b> </h2>
# <p>Apache Spark is a fast and general-purpose cluster computing system. It provides high-level APIs in <b>Java, Scala, Python and R,</b> and an optimized engine that supports general execution graphs. It also supports a rich set of higher-level tools including Spark SQL for SQL and structured data processing, MLlib for machine learning, GraphX for graph processing, and Spark Streaming.
# 
# 
# PySpark can create distributed datasets from any storage source supported by Hadoop, including your local file system,
# HDFS, Cassandra, HBase, Amazon S3, etc. 
# Spark supports text files, SequenceFiles, and any other Hadoop InputFormat.

# ## Getting the data and creating the RDD

# <h3>Initializing Spark</h3>
# <p>The first thing a Spark program must do is to create a SparkContext object, which tells Spark how to access a cluster.
# To create a SparkContext you first need to build a SparkConf object that contains information about your application.

# In[1]:

import findspark
findspark.init()
from pyspark import SparkConf, SparkContext,SQLContext
sc = SparkContext("local","App:LogisticRegression")
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
# <p> Get the testing data from web  You have data in the Data/KDD folder 

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

# In[ ]:

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

# In[ ]:

from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from time import time

# Build the model
t0 = time()
logit_model = LogisticRegressionWithLBFGS.train(training_data)
tt = time() - t0

print "Classifier trained in {} seconds".format(round(tt,3))


# ## Evaluating the model

# In order to measure the classification error on our test data, we use map on the test_data RDD and the model to predict each test point class.

# In[ ]:

predictions = logit_model.predict(test_data.map(lambda p: p.features))
predictionAndLabels = test_data.map(lambda p: p.label).zip(predictions)
t0 = time()
test_accuracy = predictionAndLabels.filter(lambda (v, p): v == p).count() / float(test_data.count())
tt = time() - t0
print "Prediction made in {} seconds. Test accuracy is {}".format(round(tt,3), round(test_accuracy,4))


# <h2>Interpreting the model</h2>

# In[ ]:

# Print the coefficients and intercept for logistic regression
print("Coefficients/Weights: " + str(logit_model.weights))
print("Intercept: " + str(logit_model.intercept))


# In[ ]:

# Save and load model
myModelPath="../myModelPath"
logit_model.save(sc, myModelPath)
sameModel = logit_model.load(sc, myModelPath)

