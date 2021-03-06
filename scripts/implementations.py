import numpy as np


### Compute grad and loss helpers

def logsig(x):
    '''Compute the log-sigmoid function component-wise based on Mächler, Martin
    paper which prevent NaN issue for instance generated by ln(exp(800)) for instance.'''
    logsig = np.zeros_like(x)
    index0 = x < -33
    logsig[index0] = x[index0]
    index1 = (x >= -33) & (x < -18)
    logsig[index1] = x[index1] - np.exp(x[index1])
    index2 = (x >= -18) & (x < 37)
    logsig[index2] = -np.log1p(np.exp(-x[index2]))
    index3 = x >= 37
    logsig[index3] = -np.exp(-x[index3])
    return logsig


def compute_loss_log(y, tx, w):
    '''Compute the loss function for a logistic model'''
    z = np.dot(tx, w)
    y = np.asarray(y)
    return np.mean((1 - y) * z - logsig(z))


def compute_sigy(x, y):
    ''' Compute sig(x)-y composent-wise
    with sig(x)=1/(1+exp(-x))'''
    index = x < 0
    result = np.zeros_like(x)
    exp_x = np.exp(x[index])
    y_index = y[index]
    result[index] = ((1 - y_index) * exp_x - y_index) / (1 + exp_x)
    exp_nx = np.exp(-x[~index])
    y_nidx = y[~index]
    result[~index] = ((1 - y_nidx) - y_nidx * exp_nx) / (1 + exp_nx)
    return result


def compute_loss_lin(y, tx, w,N): #N is the length of y
    e = np.add(y, -1 * np.matmul(tx, w))
    L = (1 / (2 *N)) * np.matmul(np.transpose(e), e)
    return L

def compute_gradient_lin(y, tx, w,N):
    e = np.add(y, -1 * np.matmul(tx, w))
    if np.isscalar(e):
        grad = (-1 / N) *e*tx
    else:
        grad = (-1 / N) * np.matmul(np.transpose(tx), e)

    return grad

def compute_gradient_log(y, tx, w):
    z = tx.dot(w)
    s = compute_sigy(z, y)
    return tx.T.dot(s) / tx.shape[0]


def compute_gradient_logridge(y, tx, w, lambda_):
    grad = compute_gradient_log(y, tx, w)
    return (grad + 2 * lambda_ * w)  # /(tx.shape[0])


### Regression functions

def least_square_GD(y, tx, initial_w, max_iter, gamma):
    w = initial_w
    n_iter = 1
    loss = 10
    loss2 = 15
    N = len(y)
    while (((n_iter < max_iter)) and (np.abs(loss2 - loss) > 1e-8)):
        loss = compute_loss_lin(y, tx, w,N)
        grad = compute_gradient_lin(y, tx, w,N)
        w = w - gamma * grad

        loss2 = compute_loss_lin(y, tx, w,N)
        n_iter = n_iter + 1
    return loss, w


def least_square_SGD(y, tx, initial_w, max_iter, gamma):
    w = initial_w
    n_iter = 1
    loss = 10
    loss2 = 15
    N = len(y)
    while (n_iter < max_iter) and (np.abs(loss2 - loss) > 1e-8):
        i = np.random.randint(0, len(y))
        loss = compute_loss_lin(y, tx, w,N)
        grad = compute_gradient_lin(y[i], tx[i], w,1)
        w = w - gamma * grad

        loss2 = compute_loss_lin(y, tx, w,N)
        n_iter = n_iter + 1
    return loss, w


def least_square(y, tx):
    N = len(y)
    #print(np.shape(tx))
    #print(np.linalg.det(tx.T@tx))
    w = np.linalg.inv(tx.T @ tx) @ tx.T @ y
    loss = compute_loss_lin(y, tx, w,N)
    return w, loss


def ridge_regression(y, tx, lambda_):
    N = len(y)
    w = np.invert(tx.T @ tx + lambda_ * np.eye(len(y))) @ tx.T @ y
    loss = compute_loss_lin(y, tx, w,N)
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    '''Same as gradient_descent_log() with ridge regularization'''
    w = initial_w
    n_iter = 1
    loss = 10
    loss2 = 15

    while (((n_iter < max_iters)) and (np.abs(loss2 - loss) > 1e-8)):
        loss = compute_loss_log(y, tx, w)
        grad = compute_gradient_log(y, tx, w)

        w = w - gamma * grad
        loss2 = compute_loss_log(y, tx, w)

        n_iter = n_iter + 1

    return loss, w


def reg_logistic_regression(y, tx, initial_w, max_iters, gamma, lambda_):
    '''Same as gradient_descent_log() with ridge regularization'''
    w = initial_w
    n_iter = 1
    loss = 10
    loss2 = 15

    while (((n_iter < max_iters)) and (np.abs(loss2 - loss) > 1e-8)):
        loss = compute_loss_log(y, tx, w)
        grad = compute_gradient_logridge(y, tx, w, lambda_)

        w = w - gamma * grad
        loss2 = compute_loss_log(y, tx, w)

        n_iter = n_iter + 1

    return loss, w
