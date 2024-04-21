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
            possible_steps = np.where(self.R[self.path[-2]] >= 0)[0]  

            q_steps = []
            for i in range(len(possible_steps)):
                q = self.Q[self.path[-2], possible_steps[i]]
                q_steps.append(q)

            # Максимальна користь від переміщення у даний стан
            max_step_benefit = np.max(q_steps)

            # Обрахунок за формулою
            self.Q[self.path[-3], self.path[-2]] = self.R[self.path[-3], self.path[-2]] + self.l_rate * max_step_benefit
            

    # Вибір дії на підставі матриці суміжності станів
    def choose_action(self):
        # Отримати доступні дії для поточного стану
        possible_actions = np.where(self.R[self.path[-1]] >= 0)[0] 
        next_index = np.random.choice(possible_actions) 
        return next_index
    

    def choose_action_based_memory(self):
        # Отримати доступні дії для поточного стану
        possible_actions = np.where(self.R[self.path[-1]] >= 0)[0] 
        # Знайти вагу переходів у кожен крок
        q_steps = []
        for i in range(len(possible_actions)):
            q = self.Q[self.path[-1], possible_actions[i]]
            q_steps.append(q)

        # Обрати потрібний крок
        index = np.argmax(q_steps) 
        return possible_actions[index]
    

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

            print(f"INFO: {iteration + 1} try done")


    # Метод для проходження поля на підставі досвіду 
    def run(self):
        self.path = [self.start_state]

        while self.path[-1] != self.goal_state:
            next_step = self.choose_action_based_memory()
            self.move(next_step)
            
        return self.path


def main():
    l_rate = 0.8
    start = 1
    end = 7

    cat = Agent(start - 1, end - 1, l_rate)

    print(f"Q matrix ")
    for row in cat.Q:
        print(" ".join(map(lambda x: f"{x}", row)))

    cat.train(10)

    print(f"Q matrix ")
    for row in cat.Q:
        print(" ".join(map(lambda x: f"{x}", row)))

    print("\n\nthe path after learning: ", cat.run())


main()
   