import numpy as np
from matrix import R, Q


class Agent:
    def __init__(self, start_state, goal_state, l_rate):
        self.start_state = start_state
        self.path = [start_state]
        self.goal_state = goal_state
        self.R = R
        self.Q = Q
        self.l_rate = l_rate

        print("Cat initialization in ", self.start_state, "\nMouse is in ", self.goal_state)


    # Оновлення матриці пам'яті з використанням навчального коефіцієнта
    def update_memory(self):
        if len(self.path) > 2:
            # Отримати доступні дії для поточного стану
            possible_steps = np.where(self.R[self.path[-1]] == 0)[0]  
            q_steps = []
            for i in range(len(possible_steps)):
                q = self.Q[self.path[-1], possible_steps[i]]
                q_steps.append(q)
            # Максимальна користь від переміщення у даний стан
            max_step_benefit = np.max(q_steps)
            # Обрахунок за формулою
            self.Q[self.path[-2], self.path[-1]] = self.R[self.path[-2], self.path[-1]] + self.l_rate * max_step_benefit


    # Вибір дії на підставі матриці суміжності станів
    def choose_action(self):
        # Отримати доступні дії для поточного стану
        possible_actions = np.where(self.R[self.path[-1]] == 0)[0] 
        next_index = np.random.choice(possible_actions) 
        return next_index
    
    
    # Переміщення в новий стан
    def move(self, action):
        self.path.append(action) 


    # Навчання інтелектуального агента
    def train(self, iterations):
        for iteration in range(iterations):
            self.path = [self.start_state]

            while self.path[-1] != self.goal_state:
                next_step = self.choose_action()
                self.move(next_step)
                self.update_memory()

            print(f"\n\nthe {iteration + 1} try: ")
            print(f"Q matrix ")
            for i in range(len(self.Q)):
                print(self.Q[i])


    # Метод для проходження поля на підставі досвіду
    # треба рухатися на основі матриця пам'яті   
    def run(self, start_state=0):
        self.path = [start_state]
        while self.path[-1] != self.goal_state:
            action = self.choose_action()
            next_state = action
            self.path.append(next_state)
            self.move(next_state)
            print(self.path)
        return self.path


def main():
    l_rate = 0.8
    start = 1
    end = 7

    cat = Agent(start - 1, end - 1, l_rate)
    cat.train(3)
    # print("the path after learning: ")
    # print(cat.run())


main()
   