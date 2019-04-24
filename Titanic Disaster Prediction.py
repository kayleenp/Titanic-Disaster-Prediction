# linear algebra
import numpy as np 

# data processing
import pandas as pd 

# data visualization
import seaborn as sns
#%matplotlib inline
from matplotlib import pyplot as plt
from matplotlib import style

# Algorithms
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB

test_df = pd.read_csv("C:\\Users\\User\\Downloads\\test.csv")
train_df = pd.read_csv("C:\\Users\\User\\Downloads\\train.csv")

#train_df.info()

# Showing survival rates of people according to sex and age
survived = 'survived'
not_survived = 'not survived'
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))
women = train_df[train_df['Sex']=='female']
men = train_df[train_df['Sex']=='male']
ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde =False)
ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False)
ax.legend()
ax.set_title('Female')
ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde = False)
ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde = False)
ax.legend()
ax.set_title('Male')

# Showing survival rates of people according to Class types
fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(5, 4))
ax = sns.barplot(x='Pclass', y='Survived', data=train_df)
ax.set_title('Survival rate of class type\n 1 = Upper \n 2 = Middle \n 3 = Lower')

# Showing survival rates of people according to Embarked location
fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(5, 4))
ax = sns.barplot(x='Embarked', y='Survived', data=train_df)
ax.set_title('Survival rate according to their embarked location\n S = Southampton \n C = Cherbourg \n Q = Queenstown')

# Showing total survivors and victim according to age
fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(6, 4))
ax = sns.swarmplot(x='Survived', y='Age', data= train_df)
ax.set_title('Total survival rate from age')

# Showing classifications of age and their fares
fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(9, 7))
ax = sns.swarmplot(x='Fare',y='Age', data=train_df)

# Showing total survivors and victim according to number of parents/ children
ax = sns.FacetGrid(data=train_df, col='Survived')
ax.height : 2
ax.map(sns.distplot, 'Parch',kde=False)
#ax.set_title('Total survivors and victims according to number of parents/ children')

# Showing total survivors and victim according to number of siblings and spouse
ax = sns.FacetGrid(data=train_df, col='Survived')
ax.height : 2
ax.map(sns.distplot, 'SibSp',kde=False)
#ax.set_title('Total survivors according to number of siblings and spouse')

# Showing total survivors and victim according to their fares
ax = sns.FacetGrid(data=train_df, col='Survived')
ax.height : 2.5
ax.aspect : 1
ax.map(sns.distplot, 'Fare',kde=False)
#ax.set_title('Total survivors according to their fares')








