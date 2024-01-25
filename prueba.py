import gymnasium as gym
import numpy as np
import torch
from neural_network import Neural_Network
# Crea el entorno Pac-Man
ENV_NAME = 'ALE/MsPacman-v5'
env = gym.make(ENV_NAME, render_mode='human',obs_type="grayscale",frameskip=1)

from utils import  add_env_wrappers


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = add_env_wrappers(env)
learning_rate = 0.0001
state_size = env.observation_space.shape
model = Neural_Network(env.observation_space.shape,env.action_space.n,learning_rate).to(device)
model.load_model("./results/pacmanTorch.pth")

action_size = env.action_space.n


# Inicia un nuevo juego

EPISODES = 3
for e in range(EPISODES):
            done = False
            score = 0
            state = env.reset()[0]
           
            lives = 3
           
           
          

            while not done:
                dead=False
               
                while not dead:
                   
                    
                    state_tensor = torch.tensor(state).unsqueeze(0).to(device)
                    prob,_ = model(state_tensor)
            
            
         
                    action = prob.sample().item()
           
                    next_state,reward,done,truncated,info, = env.step(action)  # Aplica la acción

                    state = next_state
                    
                    state = next_state
env.close()
                 
                  