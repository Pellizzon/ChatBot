class Bank:
    def __init__(self):
        self.balance = None
        self.savings = None

    def getBalance(self):
        return self.balance

    def setBalance(self, value):
        self.balance = value

    def getSavings(self):
        return self.savings

    def setSavings(self, savings):
        self.savings = savings

class GenericSwitch:
    def __init__(self):
        self.state = False

    def getState(self):
        return self.state

    def setState(self):
        self.state = not self.state

        