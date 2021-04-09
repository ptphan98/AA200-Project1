import numpy as np

def nesterov_mom(f, g, x0, n, count, prob):
    x_history = np.zeros((int(n / 2 + 1), np.size(x0)))
    x_history[0, :] = x0

    if prob == "simple1":
        alpha = .009
        beta = 0.7
    elif prob == "simple2":
        alpha = .01
        beta = 0.0001
    else:
        alpha = .001
        beta = 0.00001

    v0 = np.zeros(np.size(x0))

    index = 0
    while count() < n:
        grad = g(x0 + beta*v0)

        limit = 100
        if prob == "simple1":
            limit = 10

        abs_grad = abs(grad)
        max_grad = np.amax(abs_grad)
        if max_grad > limit:
            grad = grad*limit/max_grad

        v0 = beta*v0 - alpha*grad
        x0 = x0 + v0

        index += 1
        x_history[index, :] = x0

    return x_history

def RMSProp(f, g, x0, n, count, prob):
    alpha = .1
    gamma = 0.9
    s = np.zeros(np.size(x0))
    e = 0.00001

    x_history = np.zeros((int(n / 2 + 1), np.size(x0)))
    x_history[0, :] = x0

    index = 0
    while count() < n:
        grad = g(x0)
        s = gamma * s + (1 - gamma) * (grad * grad)
        x0 = x0 - alpha / (e + np.sqrt(s)) * grad
        index += 1
        x_history[index, :] = x0
    return x_history

def Adam(f, g, x0, n, count, prob):

    x_history = np.zeros((int(n / 2 + 1), np.size(x0)))
    x_history[0, :] = x0

    if prob == "simple1":
        alpha = .3
        gamma_v = 0.4
        gamma_s = 0.95
    else:
        alpha = 0.1
        gamma_v = 0.9
        gamma_s = 0.9

    e = 0.000001
    k = 0
    v = np.zeros(np.size(x0))
    s = np.zeros(np.size(x0))

    index = 0
    while count() < n:
        grad = g(x0)
        v = gamma_v * v + (1 - gamma_v) * grad
        s = gamma_s * s + (1 - gamma_s) * grad * grad
        k += 1
        v_hat = v / (1 - gamma_v ** k)
        s_hat = s / (1 - gamma_s ** k)
        x0 = x0 - alpha * v_hat / (e + np.sqrt(s_hat))

        index += 1
        x_history[index, :] = x0

    return x_history