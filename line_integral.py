import autograd.numpy as np
from autograd import jacobian
from scipy.integrate import quad
import matplotlib.pyplot as plt


def F(X):
    x, y = X
    return -y, -x * y

def r(t):
    return np.array([-np.sin(t), np.cos(t)])

drdt = jacobian(r)

def integrand(t):
    return F(r(t)) @ drdt(t)

I, e = quad(integrand, 0.0, np.pi / 2)
plt.scatter(I,e,color = 'red')
plt.show()

#print(f'The integral is {I:1.4f}.')