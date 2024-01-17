from environment import Environment
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

environment = Environment(mode='train',device=device)
environment.train(mode='train') 
#environment.test()         