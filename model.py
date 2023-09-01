import torch
from functional import functions as F
import math
import torch.nn as nn
import torch.nn.functional as f

class pytorch(nn.Module):

    def __init__(self, num_labels):
        super(pytorch, self).__init__()

        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64,32)
        self.fc4 = nn.Linear(32,num_labels)    

    def forward(self, x):

        x = x.view(x.shape[0], -1)

        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = f.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x

class scratch:

    def __init__(self, model_params):
        
        self.layer_config = model_params['layer_config']
        self.activation = F[model_params['activation']]
        self.output_function = F[model_params['output']]
        self.d_output_function = F["d_"+model_params['output']]
        self.d_activation = F["d_"+model_params['activation']]
        self.input_size = model_params['input_layer_size']
        self.output_size = model_params['output_layer_size']
        self.lr = model_params["learning_rate"]
        self.weight_decay = model_params["weight_decay"]

        # Total number of layers
        self.L = len(self.layer_config) + 2

        # 0 indexing convention
        self.W = [0]*(self.L)
        self.b = [0]*(self.L)
        self.a = [0]*(self.L)
        self.h = [0]*(self.L-1)

        self.del_W = [0]*(self.L)
        self.del_b = [0]*(self.L)

        layer_config = self.layer_config
        layer_config.append(self.output_size)
        layer_config.insert(0, self.input_size)
        # print(layer_config)

        for i in range(len(layer_config)-1):

            fan_in = layer_config[i]
            fan_out = layer_config[i+1]
            M = math.sqrt(6/(fan_in + fan_out))
            dist = torch.distributions.uniform.Uniform(-M, M)
            weights = dist.sample(sample_shape=(fan_out, fan_in))
            biases = dist.sample(sample_shape=(fan_out, ))
            self.W[i+1] = weights
            self.b[i+1] = biases

    # If cuda is available
    def to(self, device):

        if(device=='cuda'):

            for i in range(1, len(self.W)):

                self.W[i] = self.W[i].to(device)
                self.b[i] = self.b[i].to(device)

    def forward(self, X):

        L = self.L

        x = X.view(X.shape[0], -1)
        self.h[0] = x

        for i in range(1, L-1):

            self.a[i] = self.h[i-1] @ self.W[i].T + self.b[i]
            self.h[i] = self.activation(self.a[i])

        self.a[-1] = self.h[-1] @ self.W[-1].T + self.b[-1]
        y_hat = self.output_function(self.a[-1])

        return y_hat
    
    # Calculate gradients here
    def backward(self, y, y_hat): 

        L = self.L
       
        dW = [0]*(L)
        db = [0]*(L)

        del_L_a = self.d_output_function(y, y_hat)

        for i in range(L-1, 0, -1):

            dW[i] = torch.bmm(del_L_a.unsqueeze(dim = 2), self.h[i-1].unsqueeze(dim = 1)).mean(dim = 0)
            db[i] = del_L_a.mean(dim = 0)

            if i == 1 : break

            del_L_h = del_L_a @ self.W[i]
            del_L_a = del_L_h * self.d_activation(self.a[i-1])
            
        self.del_W = dW
        self.del_b = db

    # Take a step in the direction against the gradient
    def step(self):

        lr = self.lr
        weight_decay = self.weight_decay
        L = self.L
        
        for i in range(L):

            update_W = self.del_W[i] + 2 * weight_decay * self.W[i]
            update_b = self.del_b[i] 

            self.W[i] = self.W[i] - lr * update_W
            self.b[i] = self.b[i] - lr * update_b

model_params = {

    "loss": "cross_entropy",
    "layer_config": [500, 250, 100],
    "activation": "relu",
    "output": "softmax",
    "input_layer_size" : 28*28,
    "output_layer_size" : 10,
    "learning_rate" : 1e-2,
    "weight_decay" : 0
    
}

X = torch.randint(0,255, (4,28,28))/255
y = torch.randint(0,10, (4,))
model = scratch(model_params)
model.to('cuda')
X = X.to('cuda')
y = y.to('cuda')

output = model.forward(X)
model.backward(y, output)
model.step()