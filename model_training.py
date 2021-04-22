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

from infoClasses import Bank, GenericSwitch
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
        return input("O que você gostaria de descobrir hoje? ")
    else:
        return input(Fore.RED + "Gostaria de algo mais? " + Fore.RESET)


def dealWithInput(userInput, model, vectorizer):
    newCounts = vectorizer.transform([userInput])
    prediction = model.predict(newCounts)
    #print(Fore.GREEN + prediction[0] + Fore.RESET)
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
    frase = frase.replace("$","dinheiro")
    frase = frase.replace("foxbot", "")
    return frase


def getCityWeather():
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

def getPrecipitation():
    url = f"https://weather.com/pt-BR/clima/hoje/l/63e18eea74a484c42c3921cf52a8fec98113dbb13f6deb7c477b2f453c95b837"
    soup = BeautifulSoup(requests.get(url).content, "html.parser")

    result = soup.find("div", attrs={"class": "CurrentConditions--precipValue--RBVJT"}).text
    result = result.replace("probab.", "probabilidade")

    print(result)


def getAccountBalance(bank):
    bal = f"{bank.getBalance():,}"
    bal = bal.replace(",", ".")
    print(f"Você possui R${bal},00 na sua conta-corrente.")

def getAccountSavings(bank):
    sav = f"{bank.getSavings():,}"
    sav = sav.replace(",", ".")
    print(f"Você possui R${sav},00 na sua poupança.")


def setAirState(air):
    air.setState()
    if (air.getState()):
        print("Ligando ar-condicionado")
    else:
        print("Desligando ar-condicionado")

def setLightState(light):
    light.setState()
    if (light.getState()):
        print("Ligando luz")
    else:
        print("Desligando luz")


if __name__ == "__main__":
    luzImaginaria = GenericSwitch()
    arImaginario = GenericSwitch()

    bancoImaginario = Bank()
    bancoImaginario.setBalance(randint(0, 100_000_000))
    bancoImaginario.setSavings(randint(0, 100_000_000))
    # Load model
    with open("model_generic.sav", "rb") as f:
        vectorizer, model = pickle.load(f)
    
    with open("model_account.sav", "rb") as f:
        vectorizer_account, model_account = pickle.load(f)

    with open("model_weather.sav", "rb") as f:
        vectorizer_weather, model_weather = pickle.load(f)

    with open("model_eletro.sav", "rb") as f:
        vectorizer_eletro, model_eletro = pickle.load(f)

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
            userInputVectorized_weather, prediction_weather = dealWithInput(userInput, model_weather, vectorizer_weather)

            if prediction_weather == "Chuva":
                getPrecipitation()

            elif prediction_weather == "Temperatura":
                getCityWeather()

        elif prediction == "Consultar saldo da conta":
            userInputVectorized_account, prediction_account = dealWithInput(userInput, model_account, vectorizer_account)

            if (prediction_account == "Consultar saldo da conta-corrente"):
                getAccountBalance(bancoImaginario)

            elif (prediction_account == "Consultar saldo da poupança"):
                getAccountSavings(bancoImaginario)

        elif prediction == "Interagir com a luz ou o ar-condicionado":
            userInputVectorized_eletro, prediction_eletro = dealWithInput(userInput, model_eletro, vectorizer_eletro)

            if (prediction_eletro == "Ar-condicionado"): 
                setAirState(arImaginario)
            
            elif (prediction_eletro == "Luz"):
                setLightState(luzImaginaria)
        else:
            pass

        satisfied = isUserSatisfied()

        if satisfied == "n":
            correctionUser(model, userInputVectorized, allClasses)
        else:
            continue
