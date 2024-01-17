import collections
import gymnasium as gym
from collections import deque
import random
import numpy as np
from neural_network import Neural_Network
from replay_memory import Replay_memory
import torch
import torch.nn as nn
import torch.optim as optim
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
        self.memory = Replay_memory(capacity=5000)
       
       
        if load_model:
            #Hyperparameters for testing mode
            self.discount_factor=0.99
            self.learning_rate = 0.001
            self.epsilon = 0.2
            self.epsilon_decay = 0.99999
            self.epsilon_min = 0.1
            self.batch_size = 128
            self.train_start = 1000
          
            self.model = Neural_Network(self.state_size,self.action_size,self.learning_rate)
            self.model.load_model("./results/pacman_prueba1dqn.h5")
           

        else:
        
            #Hyperparemeters for training mode
            print("Training Mode")
            self.discount_factor=0.99
            self.learning_rate = 0.0001
            self.epsilon = 1.0
            self.epsilon_decay = 0.9999995
            self.epsilon_min = 0.1
            self.batch_size = 128
            self.train_start = 1000
            self.update_rate = 150
            self.model = Neural_Network(self.state_size,self.action_size,self.learning_rate).to(device)
            self.model_target = Neural_Network(self.state_size,self.action_size,self.learning_rate).to(device)
        
        
        #optimizer
        self.optimizer = optim.Adam(self.model.parameters(),lr=self.learning_rate)
           

 
    def get_action(self, state):
       
        if np.random.rand() <= self.epsilon:
           
            return random.randrange(self.action_size)
        else:
            
            state_tensor = torch.tensor(state).unsqueeze(0).to(self.device)
            q_value = self.model(state_tensor)
            return np.argmax(q_value[0].cpu().detach().numpy())
        
    
    #save (state,action,reward,next_state,done) in replay memory
    def append_sample(self,state,action,reward,next_state,done):
        
        exp = Experience(state,action,reward,done,next_state)
      
        self.memory.add_experience(exp)
        if self.epsilon>self.epsilon_min:
           
            self.epsilon*=self.epsilon_decay

    #train the model
    def train(self,step_counter):
        
        if self.memory.size() < self.train_start:
            return 
       
        self.optimizer.zero_grad()
        #Updating de Target model
        if step_counter % self.update_rate == 0:
            self.model_target.load_state_dict(self.model.state_dict())

        batch_size = min(self.batch_size,self.memory.size())
        batch = random.sample(self.memory.buffer,batch_size)
       
        states,actions,rewards,dones,next_states = [], [], [], [], []

        for experience in batch:
            state, action, reward, done, next_state = experience
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            next_states.append(next_state)
      
        states_tensor = torch.tensor(np.array(states,copy=False)).to(self.device)
        next_states_tensor = torch.tensor(np.array(next_states,copy=False)).to(self.device)
        actions_tensor = torch.tensor(np.array(actions,dtype=np.int64)).to(self.device)
        rewards_tensor =  torch.tensor(rewards).to(self.device)
        dones_tensor = torch.BoolTensor(dones).to(self.device)


        q_state_values = self.model(states_tensor).gather(1, actions_tensor.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            q_next_states_values = self.model_target(next_states_tensor).max(1)[0]
            q_next_states_values[dones_tensor] = 0.0
            q_next_states_values  = q_next_states_values.detach()

        target_q_val = q_next_states_values * self.discount_factor + rewards_tensor
        loss = nn.MSELoss()(q_state_values,target_q_val)
        loss.backward()
        self.optimizer.step()

     
        
    
                

