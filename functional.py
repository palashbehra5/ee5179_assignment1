import torch

def sigmoid(x):

    h = 1/(1+torch.exp(-x))
    h[h==1] = 1 - 1e-5
    h[h==0] = 1e-5

    return h

def d_sigmoid(x):

    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):

    h = x.clone()
    h[h<0] = 0

    return h

def d_relu(x):

    h = x.clone()
    h[h>=0] = 1
    h[h<0] = 0

    return h

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

    device = y.device
    k = y_hat.shape[1]

    e_l = torch.eye(k, device=device)
    e_l = e_l[y]

    res = y_hat - e_l

    return res

def cross_entropy(y, y_hat):

    k = y_hat.shape[1]
    b = y_hat.shape[0]
    device = y.device

    e_l = torch.eye(k, device=device)
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

# import matplotlib.pyplot as plt

# plt.plot(d_relu(torch.linspace(-10,10,10000)))
# plt.show()