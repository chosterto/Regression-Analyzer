import numpy as np
import matplotlib.pyplot as plt
from matrixtools import Matrix
import csv

with open('input\\inputData.csv') as f:
    inputs = list(csv.reader(f))
    points = [list(map(float, i)) for i in inputs[1:]]
    n = len(points)
    f.close()


def f(x, order, coeffs):
    y = 0
    for coeff in coeffs:
        y += coeff * (x ** order)
        order -= 1
    return y


def main():
    # WARNING... insufficient number of points may not produce a graph
    degree = int(input('Polynomial model order? '))
    x_max = float('-inf')
    x_min = float('inf')
    y_vector = Matrix([], n, 1)
    X_matrix = Matrix([], n, degree + 1)
    for x, y in points:
        if x > x_max:
            x_max = x
        if x < x_min:
            x_min = x
        y_vector.add_row([y])
        X_matrix.add_row([1.0] + [x ** i for i in range(1, degree + 1)])
    
    # p_vector contains the approximate coefficients of the polynomial 
    # that best models a set of points
    p_vector = ((X_matrix.transpose() * X_matrix).inverse() * X_matrix.transpose()) * y_vector
    coe = sum(p_vector.matrix, [])[::-1]

    # Print out results
    print('Polynomial model order =', degree, end='\n\n')
    print('f(x) = ' + ' + '.join(f'{chr(97 + i)}*x^{degree - i}' for i in range(degree + 1)), end='\n\n')
    for i in range(degree + 1):
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

    x = np.linspace(x_min, x_max, 100)

    # Plot line
    plt.plot(x, f(x, degree, coe))

    # Plot points
    plt.scatter(*list(zip(*points)))

    # Show plot
    plt.show()
    

if __name__ == '__main__':
    main()
