import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Neural_Network(nn.Module):
    def __init__(self, input_size, output_size, learning_rate):
        super(Neural_Network, self).__init__()

        # Definición de las capas
        self.convolutional_layers = nn.Sequential(
            nn.Conv2d(input_size[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3,stride=2),
            nn.ReLU()
        )
        convolutional_output = self._get_convolutional_output(input_size)
    
        self.linear_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(convolutional_output, 512), # ajusta las dimensiones según el tamaño de salida de tu última capa convolucional
            nn.ReLU(),
            nn.Linear(512, output_size)
        )



    def _get_convolutional_output(self,input_size):
      
        o = self.convolutional_layers(torch.zeros(1,*input_size))
    
        return int(np.prod(o.size()))
       
    
    def forward(self, x):
        
        x = self.convolutional_layers(x).view(x.size()[0],-1)
     
        x = self.linear_layers(x)
 
        return x

  

   

    def save_model(self, filename):
        torch.save(self.state_dict(), filename)

    def load_model(self, filename):
        self.load_state_dict(torch.load(filename))
     


