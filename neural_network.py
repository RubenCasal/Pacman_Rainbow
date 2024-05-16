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

        # Capas convolucionales
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

        # Expansión de las capas lineales incluyendo dos nuevas capas
        self.linear_layers = nn.Sequential(
            nn.Flatten(),
            NoisyLinear(convolutional_output, 1024),
            nn.ReLU(),
            NoisyLinear(1024, 512),  # Primera nueva capa lineal
            nn.ReLU(),
            NoisyLinear(512, 256),  # Primera nueva capa lineal
            nn.ReLU(),
            NoisyLinear(256, 128),  # Segunda nueva capa lineal
            nn.ReLU(),
        )

        # Actualización de las capas actor y critic para adaptarse a la última capa lineal
        self.actor_layer = nn.Sequential(
            NoisyLinear(128, output_size),
        )
        self.critic_layer = nn.Sequential(
            NoisyLinear(128, 1),
        )

    def _get_convolutional_output(self, input_size):
        # Función para calcular la salida de las capas convolucionales
        with torch.no_grad():
            sample = torch.zeros(1, *input_size)
            sample = self.convolutional_layers(sample)
        return int(np.prod(sample.size()))

    def forward(self, x):
        x = self.convolutional_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        actor = self.actor_layer(x)
        value = self.critic_layer(x)
        distribution = Categorical(F.softmax(actor, dim=-1))
        return distribution, value

    def save_model(self, filename):
        torch.save(self.state_dict(), filename)

    def load_model(self, filename):
        self.load_state_dict(torch.load(filename))

     
'''

class Neural_Network(nn.Module):
    def __init__(self, input_size, output_size,learning_rate):
        super(Neural_Network, self).__init__()

        # Nuevas capas convolucionales añadidas
        self.convolutional_layers = nn.Sequential(
            nn.Conv2d(input_size[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # Stride modificado a 2
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # Nueva capa convolucional
            nn.ReLU()
        )
        
        # Calcula la salida de las capas convolucionales para conectar con las lineales
        convolutional_output = self._get_convolutional_output(input_size)

        # Modificaciones en las capas lineales para agregar una capa adicional
        self.linear_layers = nn.Sequential(
            nn.Flatten(),
            NoisyLinear(convolutional_output, 1024),  # Aumento en la capacidad de la primera capa lineal
            nn.ReLU(),
            NoisyLinear(1024, 512),  # Nueva capa lineal
            nn.ReLU(),
        )

        # Capas actor y critic, que ahora toman entradas desde la última capa lineal
        self.actor_layer = nn.Sequential(
            NoisyLinear(512, output_size),
        )
        self.critic_layer = nn.Sequential(
            NoisyLinear(512, 1),
        )

    def _get_convolutional_output(self, input_size):
        # Función utilitaria para calcular la salida plana de las capas convolucionales
        with torch.no_grad():
            sample = torch.zeros(1, *input_size)
            sample = self.convolutional_layers(sample)
        return int(np.prod(sample.size()))

    def forward(self, x):
        x = self.convolutional_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        actor = self.actor_layer(x)
        value = self.critic_layer(x)
        distribution = Categorical(F.softmax(actor, dim=-1))
        return distribution, value

    def save_model(self, filename):
        torch.save(self.state_dict(), filename)

    def load_model(self, filename):
        self.load_state_dict(torch.load(filename))
     
 '''
