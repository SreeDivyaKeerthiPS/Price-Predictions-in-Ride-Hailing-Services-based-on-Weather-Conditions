#!/usr/bin/env python
# coding: utf-8

# # Team Name: Data Scouts
# ### Team Members:
#  
#  -  Sai Harika Paluri                                                          
#  -  Nikhil Kumar Mutyala                                              
#  -  Sree Divya Keerthi Paravasthu Siddanthi                                     
#  -  Ravi Theja Goalla   
#  
#  
# #  Project Title: Price prediction in ride-hailing services based on weather conditions
# 
# ## Project Description:
# Uber and Lyft's ride prices are not constant like public transport. They are greatly affected by the demand and supply of rides at a given time. So what exactly drives this demand? The first guess would be the time of the day; times around 9 am and 5 pm should see the highest surges on account of people commuting to work/home. Another guess would be the weather; rain/snow should cause more people to take rides.
# 
# The data is approx. for a week of Nov '18 ( I actually have included data collected while I was testing the 'querying' application so might have data spread out over more than a week. I didn't consider this as a time-series problem so did not worry about regular interval. The chosen interval was to query as much as data possible without unnecessary redundancy. So data can go from end week of Nov to few in Dec)
# The Cab ride data covers various types of cabs for Uber & Lyft and their price for the given location. You can also find if there was any surge in the price during that time. Weather data contains weather attributes like temperature, rain, cloud, etc for all the locations taken into consideration.

# #  DATA
# 
# ## Source of Data
# 
# We have acquired this dataset from  kaggle https://www.kaggle.com/ravi72munde/uber-lyft-cab-prices.

# ## Description of Data

# The dataset contains 10 features along with the price for 693071 observations. Later we add few more derived features to this dataset.<br>The date is in timestamp format. So we change that timestamp into date format, the date attribute is added to the dataset and timestamp attribute is removed. Few attributes are added like hour, fare_per_mile. The description for the 10 features is given below:<br>

#  <style>
# table {
#   font-family: arial, sans-serif;
#   border-collapse: collapse;
#   width: 100%;
#         }
# td, th {
#   border: 1px solid #dddddd;
#   text-align: left;
#   padding: 8px;
# }
# </style>
#   <table>
#   <tr>
#     <th>Feature</th>
#     <th>Description</th>
#     </tr>
#   <tr>
#     <td>Distance</td>
#     <td>Distance between source and destination of the cab ride</td>
#     </tr>
#   <tr>
#     <td>cab_type</td>
#     <td>The type of uber car or lyft booked.</td>
#     </tr>
#   <tr>
#     <td>time_stamp</td>
#     <td>Measured timestamp</td>
#     </tr>
#   <tr>
#     <td>destination</td>
#     <td>The destination of the cab ride</td>
#     </tr>
#   <tr>
#     <td>source</td>
#     <td>The source of a cab ride</td>
#     </tr>
#   <tr>
#     <td>price</td>
#     <td>Total estimated price of the cab ride.</td>
#     </tr><tr>
#     <td>name</td>
#     <td>Name of the Uber/Lyft product</td>
#     </tr>
#     <tr>
#     <td>Surge_multiplier</td>
#     <td>This is the factor by which a normal cab fare is multiplied.If it is 1 it means that there is no surge.</td>
#     </tr>
#     <tr>
#     <td>id</td>
#     <td>Unique id per ride</td>
#     </tr>
#     <tr>
#     <td>product_id</td>
#     <td>Unique id of a particular product of Lyft or Uber. </td>
#     </tr>
# </table>
# 
# The other dataset we are merging with cabs_rides is weather_data.We have 8 features and 6276 entries in this.We use time_stamp feature to merge with the cab_rides data. 
# 
# <style>
# table {
#   font-family: arial, sans-serif;
#   border-collapse: collapse;
#   width: 100%;
#         }
# td, th {
#   border: 1px solid #dddddd;
#   text-align: left;
#   padding: 8px;
# }
# </style>
#   <table>
#   <tr>
#     <th>Feature</th>
#     <th>Description</th>
#     </tr>
#   <tr>
#     <td>temp</td>
#     <td>Average temperature for a particular day</td>
#     </tr>
#   <tr>
#     <td>location</td>
#     <td>Aread in boston where the temperature was recorded.</td>
#     </tr>
#   <tr>
#     <td>clouds</td>
#     <td>Measured timestamp</td>
#     </tr>
#   <tr>
#     <td>pressure</td>
#     <td>The destination of the cab ride</td>
#     </tr>
#   <tr>
#     <td>rain</td>
#     <td>Amount of precipitation</td>
#     </tr>
#   <tr>
#     <td>time_stamp</td>
#     <td>Measured timestamp in a day</td>
#     </tr>
#     <tr>
#     <td>Humidity</td>
#     <td>Humidity percentage in the climate.</td>
#     </tr>
#     <tr>
#     <td>Wind</td>
#     <td>Wind speed recorded for a particular day</td>
#     </tr>
#     </table>
# 
# 

# # CRISM-DM PROCESS
# 
# ## Importing the dataset

# In[1]:


#Importing the neccessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn import preprocessing
import datetime
#Importing the neccessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import datetime
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split


# In[5]:


#Importing our dataset
df_cab=pd.read_csv('Datasets/cab_rides.csv')
#Creating an index label for the dataset
df_cab.reset_index(drop=True, inplace=True)


#  We need to identify the charecteristics of our dataset .For this we use three methods:-             
#  info()-Used for identifying the different features and data types in a dataset.<br>
#  shape()-We used this method to get an idea of  the rows and columns in a dataframe.<br>
#  head()-We  used this method to examine the first few rows of the dataset<br>
#  describe()-We used this method to identify the summary statistic's of all the features of the dataset.

# In[ ]:


df_cab.info()


# In[ ]:


df_cab.shape


# In[ ]:


df_cab = df_cab.iloc[0:30000, :]
df_cab.shape


# In[3]:


df_cab.head(2)


# In[4]:


df_cab.describe()


# # DATA CLEANING  AND PREPROCCESSING

# For data cleaning these were the steps that we followed:-<br>
# 1.We identified the null values in our dataset.<br>
# 2.We identified the count's of different features.<br>
# 3.We performed imputation of the null  price column using fare per pile.

# In[8]:


df_cab.isnull().sum()


# In[9]:


df_cab['cab_type'].value_counts()


# In[10]:


df_cab['destination'].value_counts()


# In[11]:


df_cab['source'].value_counts()


# In[12]:


df_cab['surge_multiplier'].value_counts()


# In[13]:


df_cab['name'].value_counts()


# We noticed that the time_stamp  attribute is not in a proper format so we converted it into a proper format.We also introduced a new attribute called fare_per_mile .This value was obtained by dividing the cab price and distance covered per ride.

# In[14]:


import datetime
#convert 13digit time stamp to datetime format
df_cab['date_time']= pd.to_datetime(df_cab['time_stamp']/1000, unit='s')
df_cab['date']= df_cab['date_time'].dt.date
df_cab['day'] = df_cab.date_time.dt.dayofweek
df_cab['hour'] = df_cab.date_time.dt.hour
#extract hours only
df_cab['fare_per_mile']= round(df_cab.price/df_cab.distance,2)


# In[15]:


#To check if cab data has changed.
df_cab.head(2)


# In[16]:


#After conversion to a new column time we can delete the exisitng column
del df_cab['time_stamp']


# We are using an imputation technique to replace the missing price values.We noticed that the missing values in the dataset are that of the type Uber-Taxi.The rate per mile in Boston for an UBER of the type 'TAXI'is a constant.The rate doesnt change irespective of the surge or weather conditions.Hence by multiplying the fare per mile and distance traveled in an Uber taxi,we were able to obtain the missing price.

# In[18]:


#Replacing null values.
df_cab['fare_per_mile']=df_cab['fare_per_mile'].astype(float)
df_cab['fare_per_mile'].fillna('2.8',inplace=True)
df_cab['price'] = df_cab['fare_per_mile']*df_cab['distance']


# In[19]:


#After replacement checking if the change is reflected in the dataset.
df_cab.price.isnull().sum()


# # Visualizations for Data Preprocessing

# Visualizations help us understand the dsitribution of the  data better.For this purpose we explored several graphs ranging from heatmaps to count plots and bar plots.

# In[20]:


#Mentioning our graph sizes
from pylab import rcParams
rcParams['figure.figsize'] = 16,10


# In[21]:


sns.heatmap(df_cab.describe()[1:].transpose(),annot=True,linecolor='w',linewidth=2,cmap=sns.color_palette('Paired'))
plt.title("DATA SUMMARY")


# In[22]:


#Heatmap from searborn library
df_corr = df_cab.corr()
sns.heatmap(df_corr, cmap =sns.color_palette("Set3"),annot = True)
plt.title("CORRELATION BETWEEN VARIABLES")


# This heatmap gave us an idea of the highly correlated variables.We observe that apart from the variable fare_per_mile .Distance and price have postive correlation whereas surge_multiper has positive correlation with price and negatve correlation with distance.

# In[23]:


#Barplot  to display Uber Vs Lyft 

plt.figure(figsize=(10,8))
flatui = [ "#3498db", "#2ecc71"]
x=['Uber','Lyft']
y = [df_cab.cab_type[(df_cab.cab_type)=='Uber'].count(),df_cab.cab_type[(df_cab.cab_type)=='Lyft'].count()]
vis1= sns.barplot(x,y,palette=flatui)


# # Data preparation for Data preprocessing

# We wanted to answer certain questions such as How does weather effect the surge? Does  it solely depend on weather only? and 
# How each weather attribute effects surge?.This is when we realised that to answer these questions we need to merge our existing cab_rides dataset with weather data for each day in Boston.We used the time_stamp attribute as the common attribute between the two datasets.<br>
# <b>NOTE:- </b><I>We use date to join both our data sets.As weather is recorded for an average day.</I>

# In[24]:


#Import the new dataset and view the attributes
df_weather=pd.read_csv('Datasets/weather.csv')
df_weather.head(2)


# In[25]:


df_weather['rain'].fillna(0, inplace = True)


# In[26]:


#Splitting the time_stamp attribute in weather to two attributes which are time and data
df_weather['date_time'] = pd.to_datetime(df_weather['time_stamp'], unit='s')
del df_weather['time_stamp']

#df_weather['date_time']


# In[27]:


#merge the datasets to refelect same time for a location
df_cab['merge_date'] = df_cab.source.astype(str) +" - "+ df_cab.date_time.dt.date.astype("str") +" - "+ df_cab.date_time.dt.hour.astype("str")
df_weather['merge_date'] = df_weather.location.astype(str) +" - "+ df_weather.date_time.dt.date.astype("str") +" - "+ df_weather.date_time.dt.hour.astype("str")
print(df_cab.head(2))
print(df_weather.head(2))


# In[28]:


df_weather = df_weather.groupby(['merge_date']).mean()
df_weather.reset_index(inplace=True)
df_weather.head()


# In[29]:


#Merging based on the date and location.
df_merged = pd.merge(df_cab, df_weather, on='merge_date')
print(df_merged.shape)


# In our newly formed merged dataset we perform info(),describe() and shape(),columns to understand the data.

# In[30]:


df_merged.info()


# In[31]:


df_merged.shape


# In[32]:


print(df_merged.columns)


# ## Visualization of merged dataset

# We have several new attributes in our merged dataset .We want visualize how these attributes contribute to our existing features such as surge,price,location cab_type and weather attributes such as rain.For visualization we have employed several plots which are catplots,countplots,boxplots,distplot and violionplot. 

# In[33]:


sns.heatmap(df_merged.describe()[1:].transpose(),annot=True,linecolor='w',linewidth=2,cmap=sns.color_palette('Paired'))
plt.title("DATA SUMMARY")


# In[34]:


df_corr = df_merged.corr()
sns.heatmap(df_corr, cmap =sns.color_palette("Set3"),annot = True)
plt.title("CORRELATION BETWEEN VARIABLES")


# In[35]:


#Catplot
sns.catplot(x="name", y="price", data=df_merged,kind="boxen", height=8, aspect=2);


# In[36]:


#Countplot
sns.countplot(df_merged['cab_type'],hue=df_merged['name'], palette='plasma')


# In[37]:


sns.boxplot(data=df_merged, x='source',y='price',palette='Blues')


# In[38]:


sns.violinplot(data=df_merged, x='destination', y='price',palette="Set3",scale="width")


# In[39]:


#Jointplot
sns.jointplot("price", "rain", data=df_merged, kind="hex",bins=15)


# In[40]:


df_merged.set_index('date').groupby('name')['price'].plot(legend=True)


# In[41]:


sns.distplot(df_merged.price, rug=True, rug_kws={"color": "g"},
                   kde_kws={"color": "k", "lw": 3, "label": "KDE"},
                   hist_kws={"histtype": "step", "linewidth": 3,
                             "alpha": 1, "color": "g"})


# ## Splitting into uber and lyft
# 
# We have decided to split the dataset to uber and lyft  so that we can generate predictions seperately .Another reason for the split was to find interesting association rules among uber and lyft data seperately.

# In[42]:


#Dropping certain columns as they are not neccessary for predictions
df_merged = df_merged.drop(['date_time','id','product_id'], axis=1)
f_merged = df_merged.drop(['fare_per_mile','surge_multiplier'],axis=1)
df_merged = df_merged.loc[:, df_merged.columns !='merge_date']


# In[43]:


#Splitting the dataset
OverallData = df_merged.drop(['cab_type'],axis=1)
uber = df_merged[df_merged['cab_type']=='Uber']
uber.reset_index(inplace=True)
uber.drop('index', axis=1, inplace=True)
lyft = df_merged[df_merged['cab_type']=='Lyft']
lyft.reset_index(inplace=True)
lyft.drop('index', axis=1, inplace=True)


# In[44]:


#Removing the cab type from uber and lyft
uber.drop('cab_type', axis=1, inplace=True)
lyft.drop('cab_type', axis=1, inplace=True)


# In[45]:


#Overview of all the columns present
print(OverallData.columns)
print(uber.columns)
print(lyft.columns)


# ### Transformations in Data Preprocessing
# 
# In this specific stage we are getting our data ready inorder to apply machine learning models.We first need to split our data into train and test set so that we can apply changes to only the relevant features.Here price is the target variable and the other features excluding time are predictor variables.
# 
# 

# In[46]:


Xd = OverallData.loc[:, OverallData.columns != 'price']
yd = OverallData['price']
Xd_train,Xd_test,yd_train,yd_test = train_test_split(Xd,yd,test_size = 0.33, random_state=42)


# In[47]:


Xu = uber.loc[:, uber.columns != 'price']
yu = uber['price']
Xu_train,Xu_test,yu_train,yu_test = train_test_split(Xu,yu,test_size = 0.33, random_state=42)


# In[48]:


Xl = lyft.loc[:, lyft.columns != 'price']
yl = lyft['price']
Xl_train,Xl_test,yl_train,yl_test = train_test_split(Xl,yl,test_size = 0.33, random_state=42)


# ## PIPELINING
# A machine learning pipeline is used to help automate machine learning workflows. They operate by enabling a sequence of data to be transformed and correlated together in a model that can be tested and evaluated to achieve an outcome, whether positive or negative.<br>
# We need to sequentially apply a list of transforms and a final estimator. Intermediate steps of pipeline must implement fit and transform methods and the final estimator only needs to implement fit.
# 
# 
# ### Standardization
# We want to apply standardization on our variables before fitting it into a model and performing prediction.The method we chose is StandarScalar.We need to make sure that the categorical variables are encoded.We use One hot encoder method for that purpose.
# 
# ### Modelling
# The dataset that we are using is a large data set with high dimensionality.We have decided to use 'RandomForest' modeling technique as it provides higher accuracy and has the power to handle a large data set with higher dimensionality. This is an Ensemble learning method.<br>
#  - As the target variable 'Price' is a continuous valued, we are using 'RandomForest Regressor' instead of RandomForest Classifier.
#  - We have decided to used 'k-fold Cross Validation' with Random Forest Regressor model for model validation (to assess the effectiveness of the model) and to mitigate overfitting in order to ensure more accurancy in the results.
#  - To measure accuracy of the predictions we are using MAE and MAPE.

# In[49]:


numerical_features = Xu.dtypes == 'float'
categorical_features = ~numerical_features
preprocess = make_column_transformer(
(numerical_features, StandardScaler()),
(categorical_features, OneHotEncoder()))
rt = RandomForestRegressor(n_estimators=200,random_state = 42)


# In[50]:


#Modelling for uber
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn import metrics

pipe = make_pipeline(preprocess, rt)
pipe.fit(Xu_train, yu_train)
yu_pred = pipe.predict(Xu_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(yu_test, yu_pred))
print('Mean Squared Error:', metrics.mean_squared_error(yu_test, yu_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(yu_test, yu_pred)))
print('Mean Absolute Percentage Error:', np.mean(np.abs((yu_test - yu_pred) / yu_test)) * 100)


# Accuracy is 100-MAPE = 98.53

# In[51]:


#Modelling for lyft
pipe = make_pipeline(preprocess, rt)
pipe.fit(Xl_train, yl_train)
yl_pred = pipe.predict(Xl_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(yl_test, yl_pred))
print('Mean Squared Error:', metrics.mean_squared_error(yu_test, yu_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(yl_test, yl_pred)))
print('Mean Absolute Percentage Error:', np.mean(np.abs((yl_test - yl_pred) / yl_test)) * 100)


# Accuracy is (100-MAPE) = 98.65

# In[52]:


#Modelling for Complete dataset
pipe = make_pipeline(preprocess, rt)
pipe.fit(Xd_train, yd_train)
yd_pred = pipe.predict(Xd_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(yd_test, yd_pred))
print('Mean Squared Error:', metrics.mean_squared_error(yd_test, yd_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(yd_test, yd_pred)))
print('Mean Absolute Percentage Error:', np.mean(np.abs((yd_test - yd_pred) / yd_test)) * 100)


# Accuracy is (100-MAPE) = 99.14

# ## Using 5-fold Cross Validation gave higher accuracy
# 
# <b> Below is the Cross validation scores for Uber, Lyft and Overall data respectively. We observe that Accuracy of the model is increased with 5-fold cross-validation </b> 

# In[53]:


from sklearn.model_selection import cross_val_score

cvs = cross_val_score(estimator = pipe, X = Xu, y = yu, cv = 5)
print("Mean Accuracy :", cvs.mean()*100)
print("Mean Standard Deviation :", cvs.std())

cvs = cross_val_score(estimator = pipe, X = Xl, y = yl, cv = 5)
print("Mean Accuracy :", cvs.mean()*100)
print("Mean Standard Deviation :", cvs.std())

cvs = cross_val_score(estimator = pipe, X = Xd, y = yd, cv = 5)
print("Mean Accuracy :", cvs.mean()*100)
print("Mean Standard Deviation :", cvs.std())


# # Association Rules
# Apriori is an algorithm for frequent item set mining and association rule learning over relational databases. It proceeds by identifying the frequent individual items in the database and extending them to larger and larger item sets as long as those item sets appear sufficiently often in the database.
# There are three major components of Apriori algorithm:
# 
# 1) Support<br>
# 2) Confidence<br>
# 3) Lift<br>
# 
# ### Support
# Support refers to the default popularity of an item and can be calculated by finding number of transactions containing a particular item divided by total number of transactions. Suppose we want to find support for item B. This can be calculated as:<br>
# <br>
# <font color="red"><center>Support(B) = (Transactions containing (B))/(Total Transactions)<br></center></font>
# 
# ### Confidence
# Confidence refers to the likelihood that an item B is also bought if item A is bought. It can be calculated by finding the number of transactions where A and B are bought together, divided by total number of transactions where A is bought. Mathematically, it can be represented as:
# <br>
# <br>
# <font color="red"><center>Confidence(A→B) = (Transactions containing both (A and B))/(Transactions containing A)<br></center></font>
# 
# ### Lift
# Lift(A -> B) refers to the increase in the ratio of sale of B when A is sold. Lift(A –> B) can be calculated by dividing Confidence(A -> B) divided by Support(B). Mathematically it can be represented as:
# <br>
# <br>
# <font color="red"><center>Lift(A→B) = (Confidence (A→B))/(Support (B))<br></center></font>
# 

# ## Uber Data

# In[108]:


uber_a=uber.copy()
print(uber_a['price'].max())
print(uber_a['price'].min())
print(uber_a['distance'].max())
print(uber_a['distance'].min())


# In[109]:


price_bins = [0,5,15,30,45,60]
price_categories = ['very low price','low price','medium range price','high price','very high price']
uber_a['price'] = pd.cut(uber_a['price'], price_bins, labels = price_categories)


# In[110]:


distance_bins = [0,1,2.5,4,5,6]
distance_categories = ['very less distance','less distance','medium range distance','far ','too far']
uber_a['distance'] = pd.cut(uber_a['distance'], distance_bins, labels = distance_categories)


# The Apriori library we are going to use requires our dataset to be in the form of a list of lists, where the whole dataset is a big list and each transaction in the dataset is an inner list within the outer big list. Currently we have data in the form of a pandas dataframe. To convert our pandas dataframe into a list of lists, execute the following script:

# In[111]:


from apyori import apriori
uber_b=uber_a[['destination','price','source','name']]
records = []
for i in range(0,16412):
    records.append([str(uber_b.values[i,j]) for j in range(0, 4)])


# ### Applying Apriori
# The next step is to apply the Apriori algorithm on the dataset. To do so, we can use the apriori class that we imported from the apyori library.
# 
# The apriori class requires some parameter values to work. The first parameter is the list of list that you want to extract rules from. The second parameter is the min_support parameter. This parameter is used to select the items with support values greater than the value specified by the parameter. Next, the min_confidence parameter filters those rules that have confidence greater than the confidence threshold specified by the parameter. Similarly, the min_lift parameter specifies the minimum lift value for the short listed rules. Finally, the min_length parameter specifies the minimum number of items that you want in your rules.

# In[112]:


rules = apriori(records, min_support=0.01, min_confidence=0.6, min_lift=4, min_length=2)


# In[113]:


for item in rules:

    # first index of the inner list
    # Contains base item and add item
    pair = item[0] 
    items = [x for x in pair]
    print("Rule: " + items[0] + " -> " + items[1])

    #second index of the inner list
    print("Support: " + str(item[1]))

    #third index of the list located at 0th
    #of the third index of the inner list

    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("******************************")


# ## Lyft Data

# In[114]:


lyft_a=lyft.copy()
print(lyft_a['price'].max())
print(lyft_a['price'].min())
print(lyft_a['distance'].max())
print(lyft_a['distance'].min())


# In[115]:


price_bins = [0,5,15,30,45,60]
price_categories = ['very low price','low price','medium range price','high price','very high price']
lyft_a['price'] = pd.cut(lyft_a['price'], price_bins, labels = price_categories)


# In[116]:


distance_bins = [0,1,2.5,4,5,6]
distance_categories = ['very less distance','less distance','medium range distance','far ','too far']
lyft_a['distance'] = pd.cut(lyft_a['distance'], distance_bins, labels = distance_categories)


# In[118]:


from apyori import apriori
lyft_b=lyft_a[['destination','price','source','name']]
records = []
for i in range(0,13455):
    records.append([str(lyft_b.values[i,j]) for j in range(0, 4)])


# In[121]:


rules = apriori(records, min_support=0.01, min_confidence=0.6, min_lift=4, min_length=2)


# In[122]:


for item in rules:

    # first index of the inner list
    # Contains base item and add item
    pair = item[0] 
    items = [x for x in pair]
    print("Rule: " + items[0] + " -> " + items[1])

    #second index of the inner list
    print("Support: " + str(item[1]))

    #third index of the list located at 0th
    #of the third index of the inner list

    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("******************************")


# We were able to generate some interesting rules in both uber and lyft.Some of them include that if your cab ride is taxi your ride price will be very low.Also some observations such as most people who go to the  North Station location prefer going in Shared Cab type.There are several other rules that were formed.However we found it interesting that some of these  rules are common for both uber and lyft data.
