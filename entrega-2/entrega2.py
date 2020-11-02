import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from tempfile import mkdtemp

def load_csv(arq):
    mysep = ";"
    mydec = ","
    myenc = "ISO-8859-1"

    df = pd.read_csv(arq, na_values=['na'],
            sep=mysep,
            decimal=mydec,
            encoding=myenc)
    return df

def load_abortion_model():
    arq = "./aborto/ep1_abortion_train.csv"
    df = load_csv(arq)

    X = df.text
    Y = df['abortion']

    pipeline = Pipeline([
        ('vect', CountVectorizer(analyzer='char_wb', ngram_range=(4, 4))),
        ('kbest', SelectKBest(k=1500)),
        ('nb', ComplementNB())
    ])

    pipeline.fit(X, Y)

    return pipeline

def load_gun_control_model():
    arq = "./controle de armas/ep1_gun-control_train.csv"
    df = load_csv(arq)

    X = df.text
    Y = df['gun-control']

    pipeline = Pipeline([
        ('vect', CountVectorizer(analyzer='char', ngram_range=(5, 10))),
        ('kbest', SelectKBest(k=9000)),
        ('nb', ComplementNB())
    ])

    pipeline.fit(X, Y)

    return pipeline

am = load_abortion_model()
gcm = load_gun_control_model()

#score abortion:  0.8946078431372549
#score gun control:  0.9338235294117647

df_test_abortion = load_csv("./entrega-2/ep1_abortion_test.csv")
df_test_gun_control = load_csv("./entrega-2/ep1_gun-control_test.csv")

abortion_position_for_abortion_dataset = am.predict(df_test_abortion.text)
gun_control_position_for_abortion_dataset = gcm.predict(df_test_abortion.text)

abortion_position_for_gun_control_dataset = am.predict(df_test_gun_control.text)
gun_control_position_for_gun_control_dataset = gcm.predict(df_test_gun_control.text)

# print(abortion_predict_am)
# print(gun_control_predict_am)

abortion_prediction = [y for x in [abortion_position_for_abortion_dataset, abortion_position_for_gun_control_dataset] for y in x] 
gun_control_prediction = [y for x in [gun_control_position_for_abortion_dataset, gun_control_position_for_gun_control_dataset] for y in x] 

result = pd.DataFrame(list(zip(abortion_prediction, gun_control_prediction)), columns =['abortion', 'gun-control']) 
result.to_csv('ep1.csv', sep=';', index=False)