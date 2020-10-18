import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from tempfile import mkdtemp

mysep = ";"
mydec = ","
myenc = "ISO-8859-1"

arq = "ep1_gun-control_train.csv"
df = pd.read_csv(arq, na_values=['na'],
        sep=mysep,
        decimal=mydec,
        encoding=myenc) #dataframe pode ser salvo para economizar memória

X = df.text
Y = df['gun-control']

pipeline = Pipeline([
        ('vect', 'passthrough'),
        #('feat', SelectKBest(score_func=chi2)),
        ('lr', LogisticRegression(random_state=123))
], memory=mkdtemp())

vect__analyzer = ['word', 'char', 'char_wb']
vect__ngram_range = [(1, 1), (1, 2), (2, 2), (1, 3), (2, 3), (3, 3)]
#feat_k = [500, 1000, 1500, 2000, 2500, 3000, 'all']
lr__max_iter = [100, 500, 1000, 2000, 3000]
lr__solver = ['lbfgs', 'liblinear']
param_grid = [
{
        'vect': [CountVectorizer()],
        'vect__analyzer': vect__analyzer,
        'vect__ngram_range': vect__ngram_range,
        #'feat__k': feat_k,
        'lr__max_iter': lr__max_iter,
        'lr__solver': lr__solver
},
{
        'vect': [TfidfVectorizer()],
        'vect__analyzer': vect__analyzer,
        'vect__ngram_range': vect__ngram_range,
        #'feat__k': feat_k,
        'lr__max_iter': lr__max_iter,
        'lr__solver': lr__solver
}]
search = GridSearchCV(pipeline, param_grid, n_jobs=-1, cv=10, scoring="f1_macro")
search.fit(X, Y)

#print(search.cv_results_)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)
print(search.best_estimator_)