import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical
from NoisyNetwork import NoisyLinear
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from NoisyNetwork import NoisyLinear
class Neural_Network(nn.Module):
    def __init__(self, input_size, output_size,learning_rate):
        super(Neural_Network, self).__init__()

        # Capas Convolucionales
        self.convolutional_layers = nn.Sequential(
            nn.Conv2d(input_size[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Salida de las capas convolucionales
        convolutional_output = self._get_convolutional_output(input_size)

        
        self.linear_layers = nn.Sequential(
            nn.Flatten(),
            NoisyLinear(convolutional_output, 1024),
            nn.ReLU(),
            NoisyLinear(1024, 512),  
            nn.ReLU(),
            NoisyLinear(512, 256),  
            nn.ReLU(),
            NoisyLinear(256, 128),  
            nn.ReLU(),
        )

        # Capa Actor
        self.actor_layer = nn.Sequential(
            NoisyLinear(128, output_size),
        )
        # Capa Crítico
        self.critic_layer = nn.Sequential(
            NoisyLinear(128, 1),
        )

    def _get_convolutional_output(self, input_size):
        # Calcular dimension de la salida de las capas convolucionales
        with torch.no_grad():
            sample = torch.zeros(1, *input_size)
            sample = self.convolutional_layers(sample)
        return int(np.prod(sample.size()))

    def forward(self, x):
        x = self.convolutional_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        # Valor Actor
        actor = self.actor_layer(x)
        # Valor Crítico
        value = self.critic_layer(x)
        # Distribución de las acciones
        distribution = Categorical(F.softmax(actor, dim=-1))
        return distribution, value

    def save_model(self, filename):
        torch.save(self.state_dict(), filename)

    def load_model(self, filename):
        self.load_state_dict(torch.load(filename))
