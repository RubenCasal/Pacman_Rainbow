from environment import Environment
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = './experimentation_models/pacmanTorch_standar_rewards.pth'
SAVE_MODEL_PATH = './experimentation_models/pacmanTorch.pth'
MODE = 'test'
GRAPH_PATH = './experimentation/pacman.png'
EPISODES_TRAIN = 12001
LEARNING_RATE = 0.0001
OPTIMIZER = 'adam' # 'adam' | 'sgd' | 'rmsprop' | 'adamw'
BUFFER_SIZE = 40000

environment = Environment(mode=MODE,device=device,model_path=MODEL_PATH,graph_path = GRAPH_PATH,save_model_path=SAVE_MODEL_PATH, EPISODES_TRAIN = EPISODES_TRAIN, learning_rate=LEARNING_RATE, optimizer= OPTIMIZER,buffer_size=BUFFER_SIZE)
environment.train(mode=MODE) 
     