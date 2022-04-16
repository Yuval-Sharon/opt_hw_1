# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def forward_sub(L,b):
    L = np.copy(L)
    b = np.copy(b)
    for i in range(len(L)):
        b[i] = b[i] / L[i,i]
        L[i][i] = 1
        for j in range(i+1,len(L)):
            b[j] = b[j] - b[i] * (L[j][i])
            L[j] = L[j] - L[i] * (L[j][i])
    return b


def backward_sub(LT,y):
    LT = np.copy(LT)
    y = np.copy(y)
    for i in reversed(range(len(LT))):
        y[i] = y[i] / LT[i][i]
        LT[i][i] = 1
        for j in range(0,i):
            y[j] = y[j] - y[i] * (LT[j][i])
            LT[j] = LT[j] - LT[i] * (LT[j][i])
    return y


def sigma_solver(s,UTb):
    s = np.copy(s)
    UTb = np.copy(UTb)
    y = np.array([UTb[i]/s[i][i] for i in range(len(UTb)-1)])
    return y

if __name__ == '__main__':
    import numpy as np

    b = np.array([6.0, 1, 5, 2])
    A = np.array([[2.0, 1, 2], [1, -2, 1], [1, 2, 3], [1, 1, 1]])
    ATB = A.transpose() @ b
    Q, R = np.linalg.qr(A)
    y = forward_sub(R.transpose(), ATB)
    x_qr = backward_sub(R, y)
    print(f'The vector that we found using qr factorization is x={x_qr}')
    u, s, vt = np.linalg.svd(A, full_matrices=True)
    # follow the equation : s@vt@x = u^t @ b
    s = np.diag(s)
    st = np.vstack((s, [0, 0, 0]))
    # y = forward_sub(st, u.transpose() @ b)
    y = sigma_solver(s,u.transpose() @ b)
    x_svd = vt.transpose() @ y
    print(f'The vector that we found using svd decomposition??? is x={x_svd}')

    W = np.diag([100, 1, 1, 1])
    ATWA = A.transpose() @ W @ A
    x_wighted = np.linalg.inv(ATWA) @ A.transpose() @ W @ b
    print(A @ x_svd - b)
    print(A @ x_wighted - b)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
