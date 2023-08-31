import torch

def sigmoid(x):

    return 1/(1+torch.exp(-x))

def d_sigmoid(x):

    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):

    x[x<0] = 0

    return x

def d_relu(x):

    x[x>=0] = 1
    x[x<0] = 0

    return x

def tanh(x):

    return 2*sigmoid(2*x) - 1

def d_tanh(x):

    return 1 - (tanh(x)**2)

def softmax(x):

    h = (torch.max(x, dim = 1, keepdim = True)[0]) - x
    h = torch.exp(h)

    sum = torch.sum(h, dim = 1, keepdim = True)

    h/=sum

    return h

# Not Really!!!
# More like d_cross_entropy_output_layer
# Output layer activation function is softmax
def d_softmax(y, y_hat):

    k = y_hat.shape[1]

    e_l = torch.eye(k)
    e_l = e_l[y]

    res = -(e_l - y_hat)

    return res

def cross_entropy(y, y_hat):

    k = y_hat.shape[1]
    b = y_hat.shape[0]

    e_l = torch.eye(k)
    e_l = e_l[y]

    log_y_hat = torch.log(y_hat)
    sum = - (e_l * log_y_hat).sum()

    return sum/b


# y = torch.randint(0,10,(4,))
# y_hat = torch.rand((4,10))
# print(cross_entropy(y, y))

functions = {
    "sigmoid": sigmoid,
    "d_sigmoid": d_sigmoid,
    "relu": relu,
    "d_relu": d_relu,
    "tanh": tanh,
    "d_tanh": d_tanh,
    "softmax": softmax,
    "d_softmax": d_softmax,
    "cross_entropy": cross_entropy
}