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

# Bon en fait visiblement ça marche pas comme je voulais, utiliser la fonction get_dummies de panda (https://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html)
# Et visiblement on pouvait pas modifier directement le dataframe de base, il fallait mettre le résultat dans un nouveau
# Mais voilà je pense que ça sera mieux de faire l'apprentissage avec ça
training_df_dummy = pd.get_dummies(training_df[cols])
testing_df_dummy = pd.get_dummies(testing_df[cols])
data_dummy = [training_df_dummy, testing_df_dummy]

# Comme ça on a direct traing_df_dummy sans Survived, et une Series target
target = training_df.Survived

## Comparaison de classifiers, en s'aidant de https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
## Et en utilisant train_test_split, classifiers importes en debut de fichier

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

score_dict = {}
for i in range(len(names)):
    score_dict[names[i]] = 0.0

X_train, X_test, y_train, y_test = train_test_split(training_df_dummy, target, test_size = .33)

for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)
    score_dict[name] = clf.score(X_test, y_test)
    
print(score_dict)