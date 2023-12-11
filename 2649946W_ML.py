#!/usr/bin/env python
# coding: utf-8

# 2649946w https://github.com/2649946w/2649946w.git

# Task 1

# In[ ]:


import sys # importing the package sys which lets you talk to your computer system.

assert sys.version_info >= (3, 7) #versions are expressed a pair of numbers (3, 7) which is equivalent to 3.7. 


# In[ ]:


from packaging import version #import the package "version"
import sklearn # import scikit-learn

assert version.parse(sklearn.__version__) >= version.parse("1.0.1") 


# In[ ]:


import matplotlib.pyplot as plt

plt.rc('font', size=14) #general font size
plt.rc('axes', labelsize=14, titlesize=14) #font size for the titles of x and y axes
plt.rc('legend', fontsize=14) # font size for legends
plt.rc('xtick', labelsize=10) # the font size of labels for intervals marked on the x axis
plt.rc('ytick', labelsize=10) # the font size of labels for intervals marked on the y axis


# In[ ]:


from pathlib import Path

IMAGES_PATH = Path() / "images" / "classification"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


# #### Markdown Answers
# 1a.
# 2a.
# 3a.
# 4.a

# TASK 2

# In[ ]:


from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

def load_housing_data(): #defines a function that loads the housing data available as .tgz file on a github URL
    tarball_path = Path("datasets/housing.tgz") # where you will save your compressed data
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True) #create datasets folder if it does not exist
        url = "https://github.com/ageron/data/raw/main/housing.tgz" # url of where you are getting your data from
        urllib.request.urlretrieve(url, tarball_path) # gets the url content and saves it at location specified by tarball_path
        with tarfile.open(tarball_path) as housing_tarball: # opens saved compressed file as housing_tarball
            housing_tarball.extractall(path="datasets") # extracts the compressed content to datasets folder
    return pd.read_csv(Path("datasets/housing/housing.csv")) #uses panadas to read the csv file from the extracted content

housing = load_housing_data() #runsthe function defined above


# In[ ]:


import pandas as pd
from pathlib import Path

housing = pd.read_csv(Path("datasets/housing/housing.csv"))


# In[ ]:


housing.info()


# In[ ]:


housing["ocean_proximity"].value_counts() # tells you what values the column for `ocean_proximity` can take


# In[ ]:


housing.hist(bins=50, figsize=(12, 8))
save_fig("attribute_histogram_plots")  # extra code
plt.show()


# In[ ]:


housing.describe()


# In[ ]:


from sklearn.datasets import fetch_openml
import pandas as pd

mnist = fetch_openml('mnist_784', as_frame=False, parser='auto')

#mnist_dataframe = pd.DataFrame(data=mnist.data, columns=mnist.feature_names)


# In[ ]:


print(mnist.DESCR)


# Discuss with your peer group what kind of information about housing in a district you think would help predict the median housing price in the district.
# - Discuss how these decisions might depend on geographical and/or cultural differences and how the information you collect would already bias the data. 
# 
# 1a. Regression would be better suited to a machine focused on housing prices as it allows for deviance where as classification as far less wiggle room in that sense.
# 2a. Classification is far more effective for hand written digit as it suits image recognition better
# 3a. Income, Average Age, Average Property Type
# 4a. Place like Glasgow are far more Flat focused whereas the state of Texas focuses on larger houses. This is dute to culture and geo-development

# In[ ]:


mnist.keys()


# In[ ]:


# cell for python code 

images = mnist.data
categories = mnist.target

# insert lines below to print the shape of images and to print the categories.


# In[ ]:


#extra code to visualise the image of digits

import matplotlib.pyplot as plt

## the code below defines a function plot_digit. The initial key work `def` stands for define, followed by function name.
## the function take one argument image_data in a parenthesis. This is followed by a colon. 
## Each line below that will be executed when the function is used. 
## This cell only defines the function. The next cell uses the function.

def plot_digit(image_data): # defines a function so that you need not type all the lines below everytime you view an image
    image = image_data.reshape(28, 28) #reshapes the data into a 28 x 28 image - before it was a string of 784 numbers
    plt.imshow(image, cmap="binary") # show the image in black and white - binary.
    plt.axis("off") # ensures no x and y axes are displayed


# In[ ]:


# visualise a selected digit with the following code

some_digit = mnist.data[4]
plot_digit(some_digit)
plt.show()


# TASK 3

# In[ ]:


from sklearn.model_selection import train_test_split

tratio = 0.2 #to get 20% for testing and 80% for training

train_set, test_set = train_test_split(housing, test_size=tratio, random_state=42) 
## assigning a number to random_state means that everytime you run this you get the same split, unless you change the data.


# In[ ]:


# extra code – shows another way to estimate the probability of bad sample

import numpy as np

sample_size = 1000
ratio_female = 0.511

np.random.seed(42)

samples = (np.random.rand(100_000, sample_size) < ratio_female).sum(axis=1)
((samples < 485) | (samples > 535)).mean()


# In[ ]:


import numpy as np
import pandas as pd

housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])


# In[ ]:


from sklearn.model_selection import train_test_split

tratio = 0.2 #to get 20% for testing and 80% for training

strat_train_set, strat_test_set = train_test_split(housing, test_size=tratio, stratify=housing["income_cat"], random_state=42)


# In[ ]:


strat_test_set["income_cat"].value_counts() / len(strat_test_set) #Prints out in order of the highest proportion first.


# 1 Each image is 20x20
# 2 Yes they are justified as they have refined the training data and therefore improved the overall accuracy of the model. As stated, SD-3 was far more usuable and therefore adjusting SD-1 to allow it to have the same kind of readability as SD-3 will improve the dataset overall

# In[ ]:


type(mnist.data)


# In[ ]:


X_train = mnist.data[:60000]
y_train = mnist.target[:60000]

X_test = mnist.data[60000:]
y_test = mnist.target[60000:]


# Task 4

# In[ ]:


housing = strat_train_set.copy()


# In[ ]:


corr_matrix = housing.corr(numeric_only=True) # argument is so that it only calculates for numeric value features
corr_matrix["median_house_value"].sort_values(ascending=False)


# In[ ]:


from pandas.plotting import scatter_matrix

features = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[features], figsize=(12, 8))
#save_fig("scatter_matrix_plot")  

#The line above is extra code you can uncomment (remove the hash at the begining) to save the image.
#But, to use this, make sure you ran the code at the beginning of this notebook defining the save_fig function

plt.show()


# In[ ]:


housing = strat_train_set.drop("median_house_value", axis=1) ## 1)
housing_labels = strat_train_set["median_house_value"].copy() ## 2)


# In[ ]:


housing.info()


# In[ ]:


housing_option3 = housing.copy() #This makes a copy of the data to variable housing_option1, so that we don't mess up the original data.

median = housing["total_bedrooms"].median() # calculating mean of the value for total_bedrooms to use in filling missing values
housing_option3["total_bedrooms"].fillna(median, inplace=True)  # option 3 - filling missing values with the median

housing_option3.info()


# In[ ]:


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median") # initialises the imputer

housing_num = housing.select_dtypes(include=[np.number]) ## includes only numeric features in the data

imputer.fit(housing_num) #calculates the median for each numeric feature so that the imputer can use them

housing_num[:] = imputer.transform(housing_num) # the imputer uses the median to fill the missing values and saves the result in variable X


# In[ ]:


housing_num.describe()


# In[ ]:


from sklearn.preprocessing import MinMaxScaler # get the MinMaxScaler

min_max_scaler = MinMaxScaler(feature_range=(-1, 1)) # setup an instance of a scaler
housing_num_min_max_scaled = min_max_scaler.fit_transform(housing_num)# use the scaler to transform the data housing_num


# In[ ]:


from sklearn.preprocessing import StandardScaler

std_scaler = StandardScaler()
housing_num_std_scaled = std_scaler.fit_transform(housing_num)


# In[ ]:


housing_num[:]=std_scaler.fit_transform(housing_num)


# In[ ]:


from sklearn.preprocessing import StandardScaler #This line is not necessary if you ran this prior to running this cell. 
#We are however including it here for completeness sake.

target_scaler = StandardScaler() #instance of Scaler
scaled_labels = target_scaler.fit_transform(housing_labels.to_frame()) #calculate the mean and standard deviation and use it to transform the target labels.


# In[ ]:


from sklearn.linear_model import LinearRegression #get the library from sklearn.linear model

model = LinearRegression() #get an instance of the untrained model
model.fit(housing_num, scaled_labels)
#model.fit(housing[["median_income"]], scaled_labels) #fit it to your data
#some_new_data = housing[["median_income"]].iloc[:5]  # pretend this is new data

#scaled_predictions = model.predict(some_new_data)
#predictions = target_scaler.inverse_transform(scaled_predictions)


# In[ ]:


some_new_data = housing_num.iloc[:5] #pretend this is new data
#some_new_data = housing[["median_income"]].iloc[:5]  # pretend this is new data

scaled_predictions = model.predict(some_new_data)
predictions = target_scaler.inverse_transform(scaled_predictions)


# In[ ]:


print(predictions, housing_labels.iloc[:5])


# In[ ]:


from sklearn.model_selection import cross_val_score

rmses = -cross_val_score(model, housing_num, scaled_labels,
                              scoring="neg_root_mean_squared_error", cv=10)


# In[ ]:


from sklearn.model_selection import cross_val_score

rmses = -cross_val_score(model, housing_num, scaled_labels,
                              scoring="neg_root_mean_squared_error", cv=10)


# In[ ]:


import tensorflow as tf

mnist = tf.keras.datasets.mnist.load_data()


# In[ ]:


print(type(mnist))


# In[ ]:


(X_train_full, y_train_full), (X_test, y_test) = mnist 
# (X_train_full, y_train_full) is the 'tuple' related to `a` and (X_test, y_test) is the 'tuple' related to `b`.
# X_train_full is the full training data and y_train_full are the corresponding labels 
# - labels indicate what digit the image is of, for example 5 if it is an image of a handwritten 5.


# In[ ]:


X_train_full = X_train_full / 255.
X_test = X_test / 255.


# In[ ]:


X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]


# In[ ]:


import numpy as np # you won't need to run this line if you ran it before in this notebook. But for completeness.

X_train = X_train[..., np.newaxis] #adds a dimension to the image training set - the three dots means keeping everything else the same.
X_valid = X_valid[..., np.newaxis]
X_test = X_test[..., np.newaxis]


# In[ ]:


tf.keras.backend.clear_session()

tf.random.set_seed(42)
np.random.seed(42)

# Unlike scikit-learn, with tensorflow and keras, the model is built by defining each layer of the neural network.
# Below, everytime tf.keras.layers is called it is building in another layer

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=3, padding="same", activation="relu", kernel_initializer="he_normal"),
    tf.keras.layers.Conv2D(64, kernel_size=3, padding="same", activation="relu", kernel_initializer="he_normal"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(128, activation="relu", kernel_initializer="he_normal"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation="softmax")
])
model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", 
              metrics=["accuracy"])

model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))


# In[ ]:


model.evaluate(X_test, y_test)


# In[ ]:


from sklearn.datasets import fetch_openml
import pandas as pd

mnist = fetch_openml('mnist_784', as_frame=False, parser='auto')

# getting the data and the categories for the data
images = mnist.data
categories = mnist.target


# In[ ]:


from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score

sgd_clf = SGDClassifier(random_state=42)

#cross validation on training data for fit accuracy

accuracy = cross_val_score(sgd_clf, images, categories, cv=10)

print(accuracy)


# Task 5: Reflection
# That's it! You've reviewed the machine learning workflow. Before you go, let's reflect on a few things together to fill in the gaps!
# 
# Task 5-1: Reflecting on the Machine Learning Workflow
# Get together with your peer group. For the following tasks, you are expected to write a markdown cell describing the workflow required. You are free to include code, but no Python code is required. Discuss the following:
# 
# What would you need to do for your code if:
# Your were to use your own data (for example, discuss survey data data and p
# 
# The data would have to be given a specific and clean path file in order to allow for ease of access. The format of data presentation may have to change, different graphs or visualisation. It would also possibly require the emedding of images within the code so allowance would have to be made for that.
# hotos)?
# You were 
# changing
# Yo
# In this regard i found it difficult to predict, i'd have to spend time learning the new model and how to make an effective piece of code within a different sort of framework.
# ur model?
# Your scali
# Id probably have to decrease the test and training data sizes in order to allow for learning times to stay reasonable.
# ng method?
# Your approach to handling mi
# Id stick with imputing. It is more important to preserve data then wipe it entirely. Whilst it may effect the accuracy of the results it would enable for overall a more precise output.
# ssing data?
# What is the significance of cross
# Cross-validation helps to compare and select an appropriate model for the specific predictive modeling problem. It is easy to understand, easy to implement, and tends to have a lower bias than other methods used to count the models efficiency scores
# umerical data.

# In[ ]:




