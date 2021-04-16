import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle
from colorama import Fore

import requests
from bs4 import BeautifulSoup  # pip install beautifulsoup4

from randomBank import Bank
from random import randint


def welcomeUser(possibleClasses):
    print(
        Fore.RED
        + f"""
Olá usuário, sou o FoxBot. Você pode me fazer perguntas relacionadas à:
1. {Fore.RESET + Fore.GREEN + possibleClasses[0] + Fore.RED};
2. {Fore.RESET + Fore.GREEN + possibleClasses[1] + Fore.RED};
3. {Fore.RESET + Fore.GREEN + possibleClasses[2] + Fore.RED};

Você pode sair a qualquer momento digitando {Fore.RESET + Fore.CYAN + "tchau" + Fore.RED}.
    """
        + Fore.RESET
    )


def questionUser():
    if 0 == 0:  # TODO: pensar em algo ou só ignorar
        return input(Fore.RED + "O que você gostaria de descobrir hoje? " + Fore.RESET)
    else:
        return input(Fore.RED + "Gostaria de algo mais? " + Fore.RESET)


def dealWithInput(userInput, model, vectorizer):
    newCounts = vectorizer.transform([userInput])
    prediction = model.predict(newCounts)
    print(Fore.GREEN + prediction[0] + Fore.RESET)
    return newCounts, prediction[0]


def isUserSatisfied():
    return input(
        Fore.RED + "Você está satisfeito com o resultado anterior? [S/n] " + Fore.RESET
    )


def correctionUser(model, userInputVectorized, allClasses):
    print(
        Fore.RED
        + f"""
1. {Fore.RESET + Fore.GREEN + allClasses[0] + Fore.RED};
2. {Fore.RESET + Fore.GREEN + allClasses[1] + Fore.RED};
3. {Fore.RESET + Fore.GREEN + allClasses[2] + Fore.RED};
4. {Fore.RESET + Fore.GREEN + allClasses[3] + Fore.RED};
"""
        + Fore.RESET
    )
    while True:
        correction = input(
            Fore.RED
            + "Selecione o que esperava com esse pedido, dentre as opções acima: "
            + Fore.RESET
        )
        if correction in ["1", "2", "3", "4"]:
            correction = int(correction) - 1

            model.partial_fit(
                userInputVectorized,
                [allClasses[correction]],
                classes=allClasses,
            )

            print(Fore.CYAN + "Obrigado por melhorar o FoxBot" + Fore.RESET)
            break
        else:
            continue


def cleanText(frase):
    frase = frase.lower()
    frase = frase.replace("-", " ")
    frase = re.sub(r"[^$\w\s]", "", frase)
    frase = frase.replace("foxbot", "")
    return frase


def getCityWeater():
    # https://www.thepythoncode.com/article/extract-weather-data-python
    # https://www.geeksforgeeks.org/how-to-extract-weather-data-from-google-in-python/

    url = f"https://www.google.com/search?q=weather"
    soup = BeautifulSoup(requests.get(url).content, "html.parser")

    result = {}
    result["city"] = soup.find("span", attrs={"class": "BNeawe tAd8D AP7Wnd"}).text
    # extract temperature now
    result["temperature"] = soup.find(
        "div", attrs={"class": "BNeawe iBp4i AP7Wnd"}
    ).text
    # get the day and hour now
    otherInfo = soup.find("div", attrs={"class": "BNeawe tAd8D AP7Wnd"}).text
    # formatting data
    data = otherInfo.split("\n")
    result["time"] = data[0]
    result["sky"] = data[1]

    print(f"{result['city']}")
    print(f"Temperatura: {result['temperature']}")
    print(f"{result['time']}, {result['sky']}")


def getAccountBalance(bank):
    bal = f"{bank.getBalance():,}"
    bal = bal.replace(",", ".")
    print(f"Você possui R${bal},00 na sua conta.")


def getInteractionFuncion():
    print("TODO")


if __name__ == "__main__":

    bancoImaginario = Bank()
    bancoImaginario.setBalance(randint(0, 100_000_000))
    # Load model
    with open("chatbot_model.sav", "rb") as f:
        vectorizer, model = pickle.load(f)

    allClasses = model.classes_
    possibleClasses = [i for i in allClasses if i != "Não sei"]
    welcomeUser(possibleClasses)

    while True:
        userInput = cleanText(questionUser())
        if userInput == "tchau":
            print(Fore.YELLOW + "Até logo!" + Fore.RESET)
            exit()

        userInputVectorized, prediction = dealWithInput(userInput, model, vectorizer)

        if prediction == "Obter informações relativas ao clima":
            getCityWeater()
        elif prediction == "Consultar saldo da conta":
            getAccountBalance(bancoImaginario)
        elif prediction == "Interagir com a luz ou o ar-condicionado":
            getInteractionFuncion()
        else:
            pass

        satisfied = isUserSatisfied()

        if satisfied == "n":
            correctionUser(model, userInputVectorized, allClasses)
        else:
            continue
