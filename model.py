import torch
from functional import functions as F
import math

model_params = {

    "loss": "cross_entropy",
    "layer_config": [500, 250, 100],
    "activation": "relu",
    "output": "softmax",
    "input_layer_size" : 28*28,
    "output_layer_size" : 10
}

class scratch:

    def __init__(self, model_params):
        
        self.layer_config = model_params['layer_config']
        self.activation = F[model_params['activation']]
        self.output_function = F[model_params['output']]
        self.input_size = model_params['input_layer_size']
        self.output_size = model_params['output_layer_size']

        # Total number of layers
        self.L = len(self.layer_config) + 2

        # 0 indexing convention
        self.W = [0]*(self.L-1)
        self.b = [0]*(self.L-1)
        self.a = [0]*(self.L)
        self.h = [0]*(self.L-1)

        self.init_weights()

    def init_weights(self):

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
            self.W[i] = weights
            self.b[i] = biases

        # for i in range(self.L-1):

        #     print(self.W[i].shape)
        #     print(self.b[i].shape)

    # If cuda is available
    def to(self, device):

        if(device=='cuda'):

            for i in range(len(self.W)):

                self.W[i] = self.W[i].to(device)
                self.b[i] = self.b[i].to(device)

    def forward(self, X):

        L = self.L

        # X is expected to be of size (batch_size, flattened_vector_size)
        x = X.view(X.shape[0], -1)
        self.h[0] = x

        for i in range(1, L-1):

            self.a[i] = self.h[i-1] @ self.W[i-1].T + self.b[i-1]
            self.h[i] = self.activation(self.a[i])
            # print(self.a[i].shape, self.h[i].shape)

        self.a[L-1] = self.h[L-2] @ self.W[L-2].T + self.b[L-2]
        y_hat = self.output_function(self.a[L-1])

        # print(y_hat.shape)

        return y_hat
        
X = torch.randint(0,255, (4,28,28))/255
model = scratch(model_params)
model.to('cuda')
X = X.to('cuda')
print(model.forward(X).shape)