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
    data.Embarked.fillna(data.Embarked.mode(), inplace = True)
    
    # Completer Fare avec la moyenne
    data.Fare.fillna(data.Fare.mean(), inplace = True)
    
# Enlever PassegengerId, Cabin et Ticket

for data in df:
    data.drop(['PassengerId', 'Cabin', 'Ticket'], axis = 1)
    
# Modifications sur les colonnes

for data in df:
    # Rassemble SibSp et Parch en FamilySize
    data['FamilySize'] = data.SibSp + data.Parch + 1
    
    # Creation d'une colonne IsAlone (1 si oui 0 sinon)
    data['IsAlone'] = 1
    data.IsAlone.loc[data.FamilySize > 1] = 0  #loc donne les indices qui verifient le booleen en argument
    
    '''
    .str.split coupe le string en 2 au niveau du premier argument
    Ici il est fait 2 fois pour recuperer le titre des gens (tous les noms sont construits de la meme facon du coup c'est facile a recuperer)
    '''
    data['Title'] = data.Name.str.split(', ', expand = True)[1].str.split(".", expand = True)[0]

    
    '''
    .cut attribue des intervalles réguliers aux variables 'continues' Fare et Age, pour les potentiels decision trees, le deuxième argument donne le nombre d'intervalles voulus, j'ai pris 6 au hasard
    '''
    data['FareRange'] = pd.cut(data.Fare, 6)
    data['AgeRange'] = pd.cut(data.Age, 6)
    