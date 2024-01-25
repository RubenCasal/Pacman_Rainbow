import collections
import gymnasium as gym
from collections import deque
import random
import numpy as np

from critic_neural_network import Critic_Neural_Network
from neural_network import Neural_Network
from replay_memory import Replay_memory
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
Experience = collections.namedtuple("Experience",field_names=['state','action','reward','done','next_state'])

class RaimbowAgent():
    def __init__(self,state_size,action_size, device, load_model=False):
        #dimensions of observations
        self.state_size = state_size
      


        #gpu device
        self.device = device
       
        
        #dimensions of actions
        self.action_size = action_size
        #replay memory
        self.memory = Replay_memory(capacity=15000)
       
       
       
        
        #Hyperparemeters for training mode                      
        print("Training Mode")
        self.discount_factor=0.99
        self.learning_rate = 1e-4
        self.epsilon = 1.0
        self.epsilon_decay = 0.99997
        self.epsilon_min = 0.1
        self.batch_size = 128
        self.train_start = 15000
        self.update_rate = 1000
        self.model = Neural_Network(self.state_size,self.action_size,self.learning_rate).to(device)
        self.model_target = Neural_Network(self.state_size,self.action_size,self.learning_rate).to(device)
        self.critic = Critic_Neural_Network(self.state_size).to(device)
    
        
        #optimizer
        self.optimizer = optim.Adam(self.model.parameters(),lr=self.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(),lr=self.learning_rate)
           

    def get_action(self, state):
       
        if np.random.rand() <= self.epsilon:
           
            return random.randrange(self.action_size)
        else:
            
          
            state_tensor = torch.tensor(state).unsqueeze(0).to(self.device)
            prob,_ = self.model(state_tensor)
            
            
         
            action = prob.sample().item()
           
   
        
        return  action
    
    #save (state,action,reward,next_state,done) in replay memory
    def append_sample(self,state,action,reward,next_state,done):
        
       
        exp = Experience(state,action,reward,done,next_state)
        self.memory.add_experience(exp)
        if self.epsilon>self.epsilon_min and self.memory.size()==self.train_start:
           
            self.epsilon*=self.epsilon_decay

    #train the model
    def train(self,step_counter):
        
        if self.memory.size() < self.train_start:
            return 
       
        
       
        #Get samples
        states_tensor,actions_tensor,rewards_tensor,next_states_tensor,dones_tensor = self.memory.sample(self.batch_size,self.device)
        

       
        dist,values = self.model(states_tensor)
        values = values.squeeze()
        log_probs = dist.log_prob(actions_tensor)

        with torch.no_grad():
            _,next_values = self.model(next_states_tensor)
            next_values[dones_tensor] = 0.0
            next_values  = next_values.detach().squeeze()
            error = rewards_tensor + self.discount_factor *next_values
            
       
        advantage = error-values
        
    
        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss =  F.mse_loss(values, error)
        total_loss = actor_loss + critic_loss
    
      
        #probabilities = np.maximum(total_loss.detach().cpu()[0],0)
        #self.memory.update_priorityies(idxs,probabilities)

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

     
        
    
                

