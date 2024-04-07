import numpy as np

def function(x):
    return 5*np.sin(10*x)*np.sin(3*x)


# декодує бінарне bitsrting у число і масштабує у заданих межах
def decode(bounds, n_bits, bitsrting):
    # найбільше число, що можу бути записане у двійковій системі у 16 бітах
    largest = 2**n_bits

    substring = bitsrting[0:n_bits]
    chars = ''.join([str(s) for s in substring])
    integer = int(chars, 2)
    value = bounds[0] + (integer/largest) * (bounds[1] - bounds[0])

    return value


# відбирає k найкращих особин у наступне покоління 
def selection(pop, scores, k=3):
    # індекс для початкового вибору
    selection_ix = np.random.randint(len(pop))

    for ix in np.random.randint(0, len(pop), k-1):
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    
    return pop[selection_ix]


# генетичний оператор схрещування створює 2 дітей від 2 батьків
def crossover(p1, p2, r_cross):
    c1, c2 = p1.copy(), p2.copy()

    # схрещування відбувається, якщо рандом менший ніж швидкість кросинговеру
    if np.random.rand() < r_cross:
        pt = np.random.randint(1, len(p1)-2)

        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
    
    return [c1, c2]


# генетичний оператор мутації
def mutate(bitstring, r_mut):
    for i in range(len(bitstring)):
        if np.random.random() < r_mut:
            bitstring[i] = 1 - bitstring[i]
    
    return bitstring


def genetic_alg(function, bounds, n_bits, n_iter, pop_size, r_cross, r_mut, delta_threshold=1e-6):
    # створення популяції з рандомних особин ( особина - двійкова 16-бітова інформація)
    pop = [np.random.randint(0, 2, n_bits).tolist() for _ in range(pop_size)]

    # присвоєння початкових min і max
    best_min, best_min_eval = 0, function(decode(bounds, n_bits, pop[0]))
    prev_best_min_eval = best_min_eval
    best_max, best_max_eval = 0, function(decode(bounds, n_bits, pop[0]))
    prev_best_max_eval = best_max_eval

    # еволюція в процесі
    for gen in range(n_iter):
        decoded_value = [decode(bounds, n_bits, p) for p in pop]
        scores = [function(d) for d in decoded_value]

        # перебираємо кожну особину у популяції для изначення поточного мін/макс
        for i in range(pop_size):
            if scores[i] < best_min_eval:
                best_min, best_min_eval = pop[i], scores[i]
                print("> Gen %d: new best_min f(%s) = %f" % (gen, decoded_value[i], scores[i]))
            if scores[i] > best_max_eval:
                best_max, best_max_eval = pop[i], scores[i]
                print("> Gen %d: new best_max f(%s) = %f" % (gen, decoded_value[i], scores[i]))

        delta = np.mean(abs(prev_best_min_eval - best_min_eval) + abs(prev_best_max_eval - best_max_eval))
        if delta < delta_threshold:
            print("Convergence reached at generation %d with delta: %f" % (gen, delta))
            break
        prev_best_min_eval = best_min_eval

        # наповнення настпуного покоління
        selected = [selection(pop, scores) for _ in range(pop_size)]
        children = list()
        for i in range(0, pop_size, 2):
            p1, p2 = selected[i], selected[i+1]
            for c in crossover(p1, p2, r_cross):
                mutate(c, r_mut)
                children.append(c)
        
        pop = children

    return [best_min, best_min_eval, best_max, best_max_eval]


def main():
    bounds = [0, 4]
    n_bits = 16
    n_iter = 10
    pop_size = 500
    r_cross = 0.9
    r_mut = 1.0 / float(n_bits) # 0.0625

    best_min, score_min, best_max, score_max = genetic_alg(function, bounds, n_bits, n_iter, pop_size, r_cross, r_mut)
    decoded_min = decode(bounds, n_bits, best_min)
    decoded_max = decode(bounds, n_bits, best_max)
    print('\n\nThe MIN result is (%s) with a score of %f\nThe MAX result is (%s) with a score of %f' % (decoded_min, score_min, decoded_max, score_max))


main()