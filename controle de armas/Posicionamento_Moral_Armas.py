import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_score

mysep = ";"
mydec = ","
myenc = "ISO-8859-1"

arq = "ep1_gun-control_train.csv"
df = pd.read_csv(arq, na_values=['na'],
        sep=mysep,
        decimal=mydec,
        encoding=myenc) #dataframe pode ser salvo para economizar memória

print(df.shape)
print(df.columns)

vect = CountVectorizer()
X = vect.fit_transform(df.text)

print(X.shape)

Y = df['gun-control']

print(Y.shape)

scores = cross_val_score(
        LogisticRegression(), X, Y, cv=10, scoring="f1_macro")

print(np.mean(scores))