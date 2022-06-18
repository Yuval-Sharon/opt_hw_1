import numpy as np
import numpy.linalg as la
from scipy.sparse import spdiags
import matplotlib.pyplot as plt

# x = np.arange(0,5, 0.01)
# n = np.size(x)
# one = int(n / 5)
# f = np.zeros(x.shape)
# f[0:one] = 0.0 + 0.5*x[0:one]
# f[(one):2*one] = 0.8 - 0.2*np.log(x[100:200]);
# f[(2*one):3*one] = 0.7 - 0.3*x[(2*one):3*one];
# f[(3*one):4*one] = 0.3
# f[(4*one):(5*one)] = 0.5 - 0.1*x[(4*one):(5*one)];
# G = spdiags([-np.ones(n), np.ones(n)], np.array([0, 1]), n-1,n)
# etta = 0.1*np.random.randn(np.size(x));
# y = f + etta
# # plt.figure(); plt.plot(x,y); plt.plot(x,f); plt.show()
#
# G = A = np.array(G.toarray())
# epsilon = 0.001

def txt_to_vector():
    with open('Covid-19-USA.txt') as txt:
        return np.array([int(line.rstrip()) for line in txt])


x = np.array([i for i in range(1, 100)])
y = txt_to_vector()
f = lambda theta: np.array(theta[0] * np.exp((-1) * theta[1] * np.square(x - theta[2])))
F = lambda theta: (1 / 2) * np.square(la.norm(f(theta) - y))
# F = lambda theta: (1 / 2) * la.norm(f(theta) - y)

def get_w(i,x_k):
    return 1/(np.abs(np.inner(G[i],x_k)) + epsilon)

def IRLS(num_iter = 10 , get_w = get_w):
    # start with w = I
    # in every iter:
    # 1) Calculate G.TRANSPOSE() @ W @ G
    # 2) calculate W matrix
    W = np.identity(len(G))
    for i in range(num_iter):
        # print(f'{len(G.transpose()[0])=} {len(W)} {len(G)=} {n=} {len(y)=}')
        the_inv =la.inv(np.identity(n) + (1/2)*((G.transpose() @ W) @ G))
        x_curr = the_inv @ y
        W = np.diag([get_w(i,x_curr) for i in range(len(W))])
        # print(f'{W.shape=} {W=}')
    return x_curr





def grad_f(theta,y):
    # - J.transpose (f(theta)-y)
    x = np.array([i for i in range(1,100)])
    f_theta = f(theta)
    J_t = np.array([np.exp( (-1) * theta[1] * np.square(x- theta[2])),
           theta[0] * np.exp( (-1) * theta[1] * np.square(x- theta[2])) * (-1) * np.square(x- theta[2]),
           theta[0] * np.exp( (-1) * theta[1] * np.square(x- theta[2])) * (2 * (x - theta[2]) * theta[1])])
    # print(f'{J_t.shape=} {f.shape=}')
    return  J_t @ (f_theta - y) , J_t

def armijo_lineserach(theta_curr,F, gradient , d):
    alpha = 10; beta = 0.5; c = 1e-2;
    for i in range(100):
        left =  F(theta_curr + alpha * d)
        right = F(theta_curr) + c * alpha * (gradient @ d)
        right1 = F(theta_curr)
        right2 = c * alpha * (gradient @ d)
        if left <= right:
        # if F(theta_curr + alpha * d) <= F(theta_curr) + c * alpha * (gradient @ d) :
            print(f'armijo lineserach took {i} to find alpha')
            return alpha

        else:
            alpha = beta * alpha
    print('finished all iter didnt find alpha')
    return alpha


def SD(theta_start, grad_F,y):
    alpha = 0.5
    theta_curr = theta_start
    for i in range(100):
        curr_grad = grad_F(theta_curr,y)[0]
        alpha = armijo_lineserach(theta_curr,F, curr_grad, - (curr_grad))
        print(f'{alpha=}')
        theta_curr = theta_curr - alpha * curr_grad

    return theta_curr

def LM(theta_start, grad_F,y):
    theta_curr = theta_start
    erros = []
    for i in range(100):
        print(f'iter num {i}')
        curr_grad , J_T= grad_F(theta_curr,y)
        d = (-1) * (la.inv(J_T @ J_T.transpose() + 0.5 * np.identity(3)) @ curr_grad )
        alpha = armijo_lineserach(theta_curr,F, curr_grad, d)
        print(f'{alpha=}')
        theta_curr = theta_curr + alpha * d
        erros.append(F(theta_curr))
        # todo: change break condition ?
        if len(erros) > 2 and np.abs(erros[-1] - erros[-2]) < 1e-3:
            return theta_curr, erros

    return theta_curr,erros

sigmoid = lambda num  : 1 / (1 + np.exp(-num))

def objective_func_grad_hessian(X , y):
    c1 = y
    c2 = 1 - y
    m = len(X)
    objective_func = lambda W : ((-1)/m) * (c1 @ (np.array([np.log(sigmoid(X[i]@W)) for i in range(m)])) +
                                            c2 @ (np.array([np.log(1 - sigmoid(X[i] @ W)) for i in range(m)])))
    gradient = lambda W: (1/m) * ( X (np.array([sigmoid(X[i]@ W) - c1[i] for i in range(m)])))
    hessian = lambda W : 5 # todo : implement

    return objective_func,gradient,hessian


if __name__ == '__main__':

    # q.3
    teta_star,err_vector = LM(np.array([1000000,0.001,110]),grad_f,y)
    F_teta_star = F(teta_star)
    print(teta_star)
    print(f'{len(err_vector)=}')
    plt.semilogy([np.abs(F_teta_star - err_vector[i]) for i in range(len(err_vector))])
    plt.title("|F(theta) - F(theta_star)|")
    plt.show()
    plt.figure(); plt.plot(x,f(teta_star)); plt.plot(x,y); plt.show()


    # sol = IRLS(10)
    # plt.figure();
    # plt.plot(x, sol);
    # plt.plot(x, f);
    # plt.show()
    # print(txt_to_vector())
    # print(len(txt_to_vector()))
