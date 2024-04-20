import numpy as np
from matrix import R, Q

class Agent:
    def __init__(self, start, end, l_rate):
        self.aim = end
        self.path = [start]
        self.R = R
        self.Q = Q
        self.l_rate = l_rate


    def learn(self, shots):

        for i in range(shots):
            while self.path[-1] != self.aim:
                indexes_to_move = np.where(self.R[self.path[-1]] == 0)[0]
                next_index = np.random.choice(indexes_to_move)
                self.path.append(next_index)
                
                
                if i > 0:
                    pass

            print(self.path)
            self.path = [self.path[0]]

        
    def run(self):
        print(self.Q[0])


def main():
    l_rate = 0.8
    start = 1
    end = 25

    cat = Agent(start - 1, end - 1, l_rate)
    cat.learn(1)
    # cat.run()


main()

    