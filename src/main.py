from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matrixtools import Matrix
from math import log
import csv

path = Path(__file__).resolve().parent.parent/'input\\inputData.csv'
with open(path, 'r') as csvf:
    inputs = list(csv.reader(csvf))
    points = [list(map(float, i)) for i in inputs[1:]]
    n = len(points)
    csvf.close()


def f(x: float, order: int, coeffs: list):
    y = 0
    for coeff in coeffs:
        y += coeff * (x ** order)
        order -= 1
    return y


def main():
    # You must have at least two points
    if n < 2:
        return

    BIC_min = float('inf')
    best_model = 1
    ps = []
    k = 0
    while k < 12:
        k += 1
        y_vector = Matrix([], n, 1)
        X_matrix = Matrix([], n, k + 1)
        for x, y in points:
            y_vector.add_row([y])
            X_matrix.add_row([1.0] + [x ** i for i in range(1, k + 1)])
        
        # p_vector contains the approximate coefficients of the polynomial
        p_vector = ((X_matrix.transpose() * X_matrix).inverse() * X_matrix.transpose()) * y_vector
        ps.append(p_vector)
        # Residual (error) sum of squares
        RSS = ((y_vector.transpose() * y_vector) - (y_vector.transpose() * (X_matrix * p_vector))).matrix[0][0]
        if RSS <= 0.0:
            break
        # Bayes information criterion
        BIC_k = n * log(RSS) + (k + 2) * log(n)
        if BIC_k < BIC_min:
            BIC_min = BIC_k
            best_model = k

    coe = sum(ps[best_model - 1].matrix, [])[::-1]

    # Print out results
    print('f(x) = ' + ' + '.join(f'{chr(97 + i)}*x^{best_model - i}' for i in range(best_model + 1)), end='\n\n')
    for i in range(best_model + 1):
        print(f'{chr(97 + i)} = {coe[i]}')

    # GRAPHING
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    x, y = list(zip(*points))
    x_line = np.linspace(min(x), max(x), 100)

    # Plot line
    plt.plot(x_line, f(x_line, best_model, coe))

    # Plot points
    plt.scatter(x, y)

    # Show plot
    plt.show()
    

if __name__ == '__main__':
    main()
