from __future__ import annotations
from typing import List


class Matrix:
    def __init__(self, matrix: List[List[float]], rows: int, cols: int):
        self.matrix = [self.__zfill_list(i, cols) for i in matrix[:rows]]
        self.rows = rows
        self.cols = cols
        self.null_rows = rows - len(matrix)

        for _ in range(self.null_rows):
            self.matrix.append([0] * self.cols)


    def add_row(self, row: List[float]) -> None:
        if self.null_rows > 0:
            # All rows should be of the same size
            row = self.__zfill_list(row, self.cols)
            self.matrix[-self.null_rows] = row
            self.null_rows -= 1
    
    def __str__(self):
        return f'{self.matrix}'

    # Adding matrices
    def __add__(self, other: Matrix) -> Matrix:
        if self.rows != other.rows or self.cols != other.cols:
            return Matrix([], 0, 0)

        matrix_sum = [
            [self.matrix[i][j] + other.matrix[i][j] 
            for j in range(self.cols)] 
            for i in range(self.rows)]

        return Matrix(matrix_sum, self.rows, self.cols)


    # Subtracting matrices
    def __sub__(self, other: Matrix) -> Matrix:
        if self.rows != other.rows or self.cols != other.cols:
            return Matrix([], 0, 0)

        matrix_diff = [
            [self.matrix[i][j] - other.matrix[i][j] 
            for j in range(self.cols)] 
            for i in range(self.rows)]

        return Matrix(matrix_diff, self.rows, self.cols)
        

    # Multiplying matrices
    def __mul__(self, other: Matrix) -> Matrix:
        C = []
        n = self.rows
        m = self.cols
        p = other.cols
        
        # Return empty matrix if number of columns
        # From matrix A != number of rows from matrix B
        if m != other.rows:
            return Matrix([], 0, 0)

        for i in range(n):
            C.append([])
            for j in range(p):
                sum = 0
                for k in range(m):
                    sum += self.matrix[i][k] * other.matrix[k][j]
                C[i].append(sum)
        
        return Matrix(C, self.rows, other.cols)

    
    def transpose(self) -> Matrix:
        return Matrix([*map(list, zip(*self.matrix))], self.cols, self.rows)
    
    def inverse(self) -> Matrix:
        # Must be a square matrix
        if self.rows != self.cols:
            return Matrix([], 0, 0)
        
        # Find inverse using Gauss-Jordan method
        n = self.rows
        A = [i + [0] * n for i in self.matrix]
        inverse = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    A[i][j + n] = 1

        for i in range(n):
            for j in range(n):
                if i != j:
                    ratio = A[j][i] / A[i][i]
                    for k in range(2 * n):
                        A[j][k] -= ratio * A[i][k]
        
        for i in range(n):
            inverse.append([])
            for j in range(n, 2 * n):
                A[i][j] /= A[i][i]
                inverse[i].append(A[i][j])
        
        return Matrix(inverse, n, n)


    def __zfill_list(self, l: List[float], n: int) -> List[float]:
        return l[:n] + [0] * (n - len(l))
    