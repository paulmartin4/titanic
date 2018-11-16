import pandas as pd
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

raw_training_df = pd.read_csv('Kaggle_data/train.csv', delimiter = ',')
raw_testing_df = pd.read_csv('Kaggle_data/test.csv', delimiter = ',')

## Feature engineering

training_df = raw_training_df

#Colonne Name :

def modify_name(name):
    '''
    Modifie le nom d'un passager
    
    Modifie le nom d'un passager pour le rendre exploitable en ne gardant que
    les informations utiles (on enlève le nom en tant que tel pour ne garder 
    que 'M.', 'Mrs.' etc...)
    
    Args:
        name (string) : Le nom du passager
        
    Returns:
        (string) : L'attribut du passager (M., Mr. etc...)
    '''
    if 'Mr.' in name:
        return 'Mr.'
    elif 'Miss.' in name:
        return 'Miss.'
    elif 'Mrs.' in name:
        return 'Mrs.'
    elif 'Master.' in  name:
        return 'Master.'
    elif 'Rev.' in name:
        return 'Rev.'
    else:
        return np.nan
    
training_df.Name = training_df['Name'].apply(modify_name)

# Colonne Cabin

def fdeck(cabin):
    '''
    Retire le nom du pont du nom de la cabine
    
    Args:
        cabin (string) : Nom de la cabine
        
    Returns:
        deck (string) : Nom du pont
    '''
    if type(cabin) == str and len(cabin) < 5:
        return cabin[0]
    else:
        return np.nan
    
def fnroom(cabin):
    '''
    Retire le numéro de la cabine du nom de la cabine
    
    Args:
        cabin (string) : Nom de la cabine
        
    Returns:
        room (int) : Numéro de la chambre
    '''
    if type(cabin) == str and len(cabin) < 5:
        return cabin[1:]
    else:
        return np.nan

deck = pd.DataFrame(training_df.Cabin).applymap(fdeck)
deck.columns = ['Deck']

room = pd.DataFrame(training_df.Cabin).applymap(fnroom)
room.columns = ['Room']

training_df = pd.concat([training_df, deck, room], axis = 1)
training_df = training_df.drop(['Cabin'], axis = 1)

training_df = training_df.drop(['Ticket'], axis = 1)

## Transforamtion colonne sexe :

def modify_sex(sex):
    '''
    Transformation de la donnée pour la rendre interprétable
    
    Args:
        sex (string) : Sexe du passager
    Returns:
        O ou 1 (int)
    '''
    if sex == 'male':
        return 0
    if sex == 'female':
        return 1
    
training_df = training_df.dropna(axis = 1, how = 'any')
training_df = training_df.drop(['Pclass', 'PassengerId'], axis = 1)    

training_df.Sex = training_df['Sex'].apply(modify_sex)

# Normalisation de la colonne "Fare"
min = training_df.Fare.min()
max = training_df.Fare.max()
mean = training_df.Fare.mean()
std = training_df.Fare.std()
    
training_df.Fare = (training_df.Fare - mean)/(std)