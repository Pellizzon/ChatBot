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
    frase = frase.replace("foxbot", "")
    return frase


if __name__ == "__main__":

    # Step 1 - DF
    data = pd.read_excel("sentencas.xlsx", engine="openpyxl")

    # Step 2 - Clean DF
    data["Sentença"] = data["Sentença"].apply(cleanText)

    # Step 3 - split test and train DF
    X_train, X_test, y_train, y_test = train_test_split(
        data["Sentença"], data["Intenção"], test_size=1 / 3, random_state=42
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

    # model acc score
    print(accuracy_score(y_test, y_pred))

    # Save model
    with open("chatbot_model.sav", "wb") as fout:
        pickle.dump((vectorizer, model), fout)
