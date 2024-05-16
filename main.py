from environment import Environment
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = './experimentation_models/pacmanTorch.pth'
SAVE_MODEL_PATH = './experimentation_models/pacmanTorch.pth'
MODE = 'test'
GRAPH_PATH = './experimentation/pacman.png'

environment = Environment(mode=MODE,device=device,model_path=MODEL_PATH,graph_path = GRAPH_PATH,save_model_path=SAVE_MODEL_PATH)
environment.train(mode=MODE) 
#environment.test()         