# ex a:

# program to find and return the inverse matrix
def transpose(m):
    return list(map(list, zip(*m)))


def getMatrixMinor(m, i, j):
    return [row[:j] + row[j+1:] for row in (m[:i]+m[i+1:])]


def getMatrixDeternminant(m):
    if len(m) == 2:
        return m[0][0]*m[1][1]-m[0][1]*m[1][0]
    determinant = 0
    for c in range(len(m)):
        determinant += ((-1)**c)*m[0][c]*getMatrixDeternminant(getMatrixMinor(m, 0, c))
    return determinant


def getMatrixInverse(m):
    determinant = getMatrixDeternminant(m)
    if determinant == 0:
        print("determinant = 0 - there is no inverse matrix")
        return 0
    # special case for 2x2 matrix:
    if len(m) == 2:
        return [[m[1][1]/determinant, -1*m[0][1]/determinant],
                [-1*m[1][0]/determinant, m[0][0]/determinant]]

    # find matrix of cofactors
    cofactors = []
    for r in range(len(m)):
        cofactorRow = []
        for c in range(len(m)):
            minor = getMatrixMinor(m, r, c)
            cofactorRow.append(((-1)**(r+c)) * getMatrixDeternminant(minor))
        cofactors.append(cofactorRow)
    cofactors = transpose(cofactors)
    for r in range(len(cofactors)):
        for c in range(len(cofactors)):
            cofactors[r][c] = cofactors[r][c]/determinant
    return cofactors


def printmatrix(size, matrix):
    for i in range(size):
        print(list(matrix[i]))


# ex b:

# Program to decompose a matrix into lower and upper triangular matrix
MAX = 100


def luDecomposition(mat, n):
    lower = [[0 for x in range(n)]
             for y in range(n)]
    upper = [[0 for x in range(n)]
             for y in range(n)]

    # Decomposing matrix into Upper and Lower triangular matrix
    for i in range(n):

        # Upper Triangular
        for k in range(i, n):

            # Summation of L(i, j) * U(j, k)
            sum = 0
            for j in range(i):
                sum += (lower[i][j] * upper[j][k])

            # Evaluating U(i, k)
            upper[i][k] = mat[i][k] - sum

        # Lower Triangular
        for k in range(i, n):
            if i == k:
                lower[i][i] = 1  # Diagonal as 1
            else:

                # Summation of L(k, j) * U(j, i)
                sum = 0
                for j in range(i):
                    sum += (lower[k][j] * upper[j][i])

                # Evaluating L(k, i)
                lower[k][i] = float((mat[k][i] - sum) / upper[i][i])
    return upper, lower


def matrixmult (A, B):
    rows_A = len(A)
    cols_A = len(A[0])
    rows_B = len(B)
    cols_B = len(B[0])

    if cols_A != rows_B:
      print("Cannot multiply the two matrices. Incorrect dimensions.")
      return

    # Create the result matrix
    # Dimensions would be rows_A x cols_B
    C = [[0 for row in range(cols_B)] for col in range(rows_A)]

    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                C[i][j] += A[i][k] * B[k][j]
    return(C)


# Driver code
size = 4
print("lets scan a matrix, enter value after value")
matrix = []
for i in range(0, size):
    matrix.append([])
    for j in range(0, size):
        matrix[i].append(0)

for i in range(size):
    for j in range(size):
        matrix[i][j] = input()
        matrix[i][j] = float(matrix[i][j])

print("enter the b vector")
vectorb = []
for i in range(size):
    vectorb.append([0])
    vectorb[i][0] = (input())
    vectorb[i][0] = float(vectorb[i][0])

print(" vector b is :")
print(vectorb)
print("the original matrix is :")
printmatrix(size, matrix)
print("the inverse matrix is :")
inverse = getMatrixInverse(matrix)

if inverse != 0:
    printmatrix(size, inverse)
    U, L = luDecomposition(matrix, size)

    inverseL = getMatrixInverse(L)
    inverseU = getMatrixInverse(U)

    y = matrixmult(inverseL, vectorb)
    inverseU_mul_y = matrixmult(inverseU, y)

    print("the result vector - X vector :")
    print(inverseU_mul_y)