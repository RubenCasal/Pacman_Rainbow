import collections
import gymnasium as gym
from collections import deque
import random
import numpy as np
from Prioritized_Buffer_Replay import Prioritized_Buffer_Replay
from Prioritized_replay import Prioritized_Replay


from neural_network import Neural_Network
from replay_memory import Replay_memory
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
Experience = collections.namedtuple("Experience",field_names=['state','action','reward','done','next_state'])

class RaimbowAgent():
    def __init__(self,state_size,action_size, device,  model_path,learning_rate, optimizer_type,buffer_size, load_model=False,):
        #Dimensión de las observaciones
        self.state_size = state_size
        #gpu 
        self.device = device
        #Dimensión de las acciones
        self.action_size = action_size
        #Número de pasos de N-Step Learning
        self.n_step = 2
        #Buffer de N-Step Learning
        self.n_step_buffer = deque(maxlen=self.n_step)
        #Prioritized Experience Replay Buffer
        self.memory = Prioritized_Replay(capacity=buffer_size)
        if load_model:
            print("Testing Mode")
            self.epsilon = 0.0
            self.learning_rate = learning_rate
            self.epsilon_decay = 0.9995
            self.epsilon_min = 0.1
            self.model = Neural_Network(self.state_size,self.action_size,self.learning_rate).to(device)
            self.model.load_model(model_path)


        else:
            #Hiperparámetros para el entrenamiento                    
            print("Training Mode")
            self.discount_factor=0.99
            self.learning_rate = learning_rate
            self.epsilon = 1.0
            self.epsilon_decay = 0.9995
            self.epsilon_min = 0.01
            self.batch_size = 128
            self.train_start = buffer_size
            self.update_rate = 1000
            self.model = Neural_Network(self.state_size,self.action_size,self.learning_rate).to(device)
            self.model_target = Neural_Network(self.state_size,self.action_size,self.learning_rate).to(device)
           
            #optimizer
            if optimizer_type == 'adam':
                #self.learning_rate = 0.0001
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            elif optimizer_type == 'sgd':
                #self.learning_rate = 0.0001
                self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate,momentum=0.9)
            elif optimizer_type == 'rmsprop':
                 #self.learning_rate = 0.00025
                self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
            elif optimizer_type == 'adamw':
                #self.learning_rate = 0.0001
                self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=0.01)
            else:
                raise ValueError(f"Optimizer type '{optimizer_type}' is not supported.")


    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            #Selecciona una acción aleatoria
            return random.randrange(self.action_size)
        else:  
            state_tensor = torch.tensor(state).unsqueeze(0).to(self.device)
            prob,_ = self.model(state_tensor)
        action = prob.sample().item()   
        return  action
    
 
    def calculate_n_step_info(self):
        reward_n_step = 0
        for idx, (_, _, reward, _, _) in enumerate(self.n_step_buffer):
            #Calcula la recompensa acumulada de los n-pasos
            reward_n_step += (self.discount_factor ** idx) * reward
        _, _, _, next_state_n_step, done_n_step = self.n_step_buffer[-1]

        return reward_n_step, next_state_n_step, done_n_step
    
    
    def append_sample(self,state,action,reward,next_state,done):
        self.n_step_buffer.append((state,action,reward,next_state,done))

        if len(self.n_step_buffer)< self.n_step and not done:
            return
        
        reward_n_step, next_state_n_step, done_n_step = self.calculate_n_step_info()
        state_n_step, action_n_step = self.n_step_buffer[0][:2]
        exp = Experience(state_n_step,action_n_step,reward_n_step,done_n_step,next_state_n_step)
        state_tensor = torch.tensor(state_n_step).unsqueeze(0).to(self.device)
        next_state_tensor = torch.tensor(next_state_n_step).unsqueeze(0).to(self.device)
        dist,value = self.model(state_tensor)
       
        with torch.no_grad():
            _,next_values = self.model(next_state_tensor)
            next_values[done] = 0.0
            next_values  = next_values.detach()
            error = reward + self.discount_factor *next_values
            
       
        advantage = error-value
        advantage.detach().cpu().numpy()
        exp = Experience(state_n_step,action_n_step,reward_n_step,done_n_step,next_state_n_step)
        self.memory.add(exp,advantage)
        if self.epsilon>self.epsilon_min and self.memory.size()==self.train_start:
           
            self.epsilon*=self.epsilon_decay

    def train(self,step_counter):
        
        if self.memory.size() < self.train_start or step_counter % 10 == 0:
            return 
       
        #Samples
        idxs,states_tensor,actions_tensor,rewards_tensor,next_states_tensor,dones_tensor = self.memory.sample(self.batch_size,self.device)
        
        dist,values = self.model(states_tensor)
        values = values.squeeze()
        log_probs = dist.log_prob(actions_tensor)

        with torch.no_grad():
            _,next_values = self.model(next_states_tensor)
            next_values[dones_tensor] = 0.0
            next_values  = next_values.detach().squeeze()
            error = rewards_tensor + self.discount_factor *next_values
            
       
        advantage = error-values
        
        actor_loss = -(log_probs * advantage.detach())
        actor_loss_mean = actor_loss.mean()
        critic_loss = advantage**2
        
    
        total_loss = (actor_loss_mean + critic_loss)
        total_loss_mean = total_loss.mean()
      
        probabilities = total_loss.detach().cpu().numpy()
       
        self.memory.update_priorities(idxs,probabilities)

        self.optimizer.zero_grad()
        total_loss_mean.backward()
        self.optimizer.step()

     
        
    
                

