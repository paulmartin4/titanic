import pandas as pd
import numpy as np
import copy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.preprocessing import LabelEncoder
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

raw_training_df = pd.read_csv('kaggle_data/train.csv', delimiter = ',')
raw_testing_df = pd.read_csv('kaggle_data/test.csv', delimiter = ',')

#### Feature engineering

training_df = copy.deepcopy(raw_training_df)
testing_df = copy.deepcopy(raw_testing_df)

df = [training_df, testing_df]

## Complete Age, Embarked, Fare

for data in df:
    # Complete Age avec la moyenne
    data.Age.fillna(data.Age.mean(), inplace = True)
    
    # Complete Embarked with mode (the feature the most represented)
    data.Embarked.fillna(data.Embarked.mode()[0], inplace = True)
    
    # Complete Fare with mean
    data.Fare.fillna(data.Fare.mean(), inplace = True)
    
    
## Modify columns

for data in df:
    # Get SibSp and Parch together in FamilySize
    data['FamilySize'] = data.SibSp + data.Parch + 1
    
    data['Title'] = data.Name.str.split(', ', expand = True)[1].str.split(".", expand = True)[0]
    
    '''
    When a feature doesn't occurs many times (less than 10 times), it is not
    relevant to consider it, therefore we put the so called rare titles together
    (such as Capt or Jonkheer) in a single titale 'Other'
    '''
    
    rare_titles = data.Title.value_counts() < 10
    data.Title = data.Title.apply(lambda x : 'Other' if rare_titles.loc[x] else x)
    
    # Creation of an 'IsAlone' binary column
    data['IsAlone'] = 1
    data.IsAlone.loc[data.FamilySize > 1] = 0  #loc returns the indexes which verify the boolean given as an argument
    
    '''
    .cut separates a continuous variable (here 'Fare' and 'Age') into several intervals
    '''
    data['FareRange'] = pd.cut(data.Fare, 6)
    data['AgeRange'] = pd.cut(data.Age, 6)
    
    # Normalization of Age and Fare columns
    
    data.Fare = data.Fare.apply(lambda x : (x - data.Fare.mean())/data.Fare.std())
    data.Age = data.Age.apply(lambda x : (x - data.Age.mean())/data.Age.std())
    
# Then we can drop several irrelevant columns
training_df = training_df.drop(['Name', 'PassengerId', 'Cabin', 'Ticket'], axis = 1)
testing_df = testing_df.drop(['Name', 'PassengerId', 'Cabin', 'Ticket'], axis = 1)
    
    
## Implementation of one_hot encoding for relevant features with the get_dummies function (https://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html)

# Columns to modify
cols = ['Sex','Pclass', 'Embarked', 'Title','SibSp', 'Parch', 'Age', 'Fare', 'FamilySize', 'IsAlone']


training_df_dummy = pd.get_dummies(training_df[cols])
testing_df_dummy = pd.get_dummies(testing_df[cols])
data_dummy = [training_df_dummy, testing_df_dummy]



## Classifiers comparison, with the help of https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA", "LDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=100, max_features=1),
    MLPClassifier(max_iter = 400, alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    LinearDiscriminantAnalysis()]

score_dict = {}
for i in range(len(names)):
    score_dict[names[i]] = []

target = training_df.Survived

rounds = 20

for k in range(rounds):
    for name, clf in zip(names, classifiers):
        X_train, X_test, y_train, y_test = train_test_split(training_df_dummy, target, test_size = .33)
        clf.fit(X_train, y_train)
        score_dict[name] += [clf.score(X_test, y_test)]
        
alg_comparison = pd.DataFrame(score_dict)

## Generation of the Kaggle submission file, with the selected classifiers : Gaussian Process, MLPClassifier and LDA

sel_names = ["Gaussian Process", "Neural Net", "LDA", "Random Forest"]

sel_classifiers = [
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    MLPClassifier(max_iter = 400, alpha = 1),
    LinearDiscriminantAnalysis(),
    RandomForestClassifier(max_depth = 5, n_estimators = 100, max_features = 1)
]


X_train, y_train = training_df_dummy, target
X_test, y_test = testing_df_dummy, pd.Series(np.zeros(testing_df_dummy.shape[0]))

for k in range(rounds):
    for name, clf in zip(sel_names, sel_classifiers):
        clf.fit(X_train, y_train)
        y_test += clf.predict_proba(X_test)[:, 1]
        
y_test = y_test/(rounds * len(sel_names))

y_test = y_test.apply(lambda x : 1 if x > 0.5 else 0)

y_test = pd.DataFrame(y_test, columns = ['Survived'])

y_test['PassengerId'] = range(892, 892 + 418)

y_test.to_csv(path_or_buf = '~/Documents/titanic/titanic/submission.csv')