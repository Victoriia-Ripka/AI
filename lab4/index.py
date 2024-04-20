import numpy as np
from matrix import R, Q

class Agent:
    def __init__(self, l_rate):
        self.R = R
        self.Q = Q
        self.l_rate = l_rate


    def learn(self):
        print(self.R[0])


    def run(self):
        print(self.Q[0])


def main():
    l_rate = 0.8

    cat = Agent(l_rate)
    cat.learn()
    cat.run()


main()

    