from environment import Environment
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = './results/pacmanTorch.pth'
environment = Environment(mode='test',device=device,model_path=model_path)
environment.train(mode='test') 
#environment.test()         