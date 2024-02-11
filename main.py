from environment import Environment
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = './results/pacmanTorch_19_rainbow.pth'
environment = Environment(mode='train',device=device,model_path=model_path)
environment.train(mode='train') 
#environment.test()         