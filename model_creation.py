import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle


def cleanText(frase):
    frase = frase.lower()
    frase = frase.replace("-", " ")
    frase = re.sub(r"[^$\w\s]", "", frase)
    frase = frase.replace("$","dinheiro")
    frase = frase.replace("foxbot", "")
    return frase


def modelGenerator(data):
    # Step 2 - Clean DF
    data["Sentença"] = data["Sentença"].apply(cleanText)

    # Step 3 - split test and train DF
    X_train, X_test, y_train, y_test = train_test_split(
        data["Sentença"], data["Intenção"], test_size=1 / 3
    )


    # Step 4 - naive-Bayes classifier
    vectorizer = CountVectorizer()
    counts = vectorizer.fit_transform(X_train)

    model = MultinomialNB()
    model.fit(counts, y_train)

    """
    #Probabilidade para cada classe
    #model.predict_proba(counts_da_nova_sentenca)
    """

    counts_da_nova_sentenca = vectorizer.transform(X_test)
    
    y_pred = model.predict(counts_da_nova_sentenca)


    # # model acc score
    print(accuracy_score(y_test, y_pred))

    return model, vectorizer
    




if __name__ == "__main__":

    # Step 1 - DF
    #First choice
    data_generic = pd.read_excel("sentencas.xlsx", engine="openpyxl")

    #Sub choices
    data_weather = pd.read_excel("sentencas_clima.xlsx", engine="openpyxl")
    data_eletro = pd.read_excel("sentencas_eletro.xlsx", engine="openpyxl")
    data_account = pd.read_excel("sentencas_conta.xlsx", engine="openpyxl")

    # datas = [data_generic, data_weather, data_eletro, data_account]

    # print(list(map(modelGenerator, datas)))

    #Models and vectorizers
    model_generic, vectorizer_generic = modelGenerator(data_generic)

    model_weather, vectorizer_weather = modelGenerator(data_weather)
    model_eletro, vectorizer_eletro = modelGenerator(data_eletro)
    model_account, vectorizer_account = modelGenerator(data_account)

    #Lists
    vectorizers = [vectorizer_generic, vectorizer_weather, vectorizer_eletro, vectorizer_account]
    models = [model_generic, model_weather, model_eletro, model_account]
    names = ["model_generic", "model_weather", "model_eletro", "model_account"]

    # Save model
    for i in range(len(models)):
        with open(""+names[i]+".sav", "wb") as fout:
            pickle.dump((vectorizers[i], models[i]), fout)
