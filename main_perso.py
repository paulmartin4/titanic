import pandas as pd
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

raw_training_df = pd.read_csv('kaggle_data/train.csv', delimiter = ',')
raw_testing_df = pd.read_csv('kaggle_data/test.csv', delimiter = ',')

## Feature engineering

training_df = raw_training_df
testing_df = raw_testing_df

df = [training_df, testing_df]

# Completer Age, Embarked, Fare

for data in df:
    # Completer Age avec la moyenne
    data.Age.fillna(data.Age.mean(), inplace = True)
    
    # Completer Embarked avec mode (la feature la plus présente)
    data.Embarked.fillna(data.Embarked.mode()[0], inplace = True)
    
    # Completer Fare avec la moyenne
    data.Fare.fillna(data.Fare.mean(), inplace = True)
    
# Enlever PassegengerId, Cabin et Ticket

for data in df:
    data = data.drop(['PassengerId', 'Cabin', 'Ticket'], axis = 1)
    
# Modifications sur les colonnes

for data in df:
    # Rassemble SibSp et Parch en FamilySize
    data['FamilySize'] = data.SibSp + data.Parch + 1
    
    data['Title'] = data.Name.str.split(', ', expand = True)[1].str.split(".", expand = True)[0]
    
    # Creation d'une colonne IsAlone (1 si oui 0 sinon)
    data['IsAlone'] = 1
    data.IsAlone.loc[data.FamilySize > 1] = 0  #loc donne les indices qui verifient le booleen en argument
    
    '''
    .cut attribue des intervalles réguliers aux variables 'continues' Fare et Age, pour les potentiels decision trees, le deuxième argument donne le nombre d'intervalles voulus, j'ai pris 6 au hasard
    '''
    data['FareRange'] = pd.cut(data.Fare, 6)
    data['AgeRange'] = pd.cut(data.Age, 6)
    
# Du coup on peut tej Name
for data in df:
    data = data.drop(['Name'], axis = 1)
    
# Askip quand une feature est pas presente plus de 10 fois, ça sert pas à grand chose de la garder
# Donc Tous les titres presents moins de 10 fois sont regroupés dans Other
# Après les avoir localises avec loc
# Avec une petite fonction lambda parce qu'on est des hackers de l'extreme

for data in df:
    rare_titles = data.Title.value_counts() < 10
    data.Title = data.Title.apply(lambda x : 'Other' if rare_titles.loc[x] else x)
    
# Début de l'idée de Paul avec les features prenant plus de 2 valeurs differentes et les colonnes de 0 et 1, sauf pour Pclass, FamilySize parce que je pense que c'est pas con de laisser ça comme ça
# Y'a LabelEncoder de skleanr qui fait ça https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for data in df:
    data['SexeCode'] = le.fit_transform(data.Sex)
    data['Embarked_Code'] = le.fit_transform(data.Embarked)
    data['Title_Code'] = le.fit_transform(data.Title)
    data['FareRange_Code'] = le.fit_transform(data.FareRange)
    data['AgeRange_Code'] = le.fit_transform(data.AgeRange)
