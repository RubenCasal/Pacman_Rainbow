from collections import deque
import random
import numpy as np
import torch
class Replay_memory:
    def __init__(self,capacity):
        self.buffer = deque(maxlen=capacity)


    #add experience 
    def add_experience(self,experience):
        
        self.buffer.append(experience)

    def size(self):
        return len(self.buffer)
    
    def clear(self):
        self.buffer.clear()


  

   


    def sample(self,batch_size,device):

        batch_size = min(batch_size,self.size())
        experiences = random.sample(self.buffer, batch_size)

       

        states = torch.tensor(np.array([e.state for e in experiences if e is not None])).to(device)
        next_states = torch.tensor(np.array([e.next_state for e in experiences if e is not None])).to(device)
        actions = torch.tensor(np.array([e.action for e in experiences if e is not None],dtype=np.int64)).to(device)
        rewards=  torch.tensor([e.reward for e in experiences if e is not None]).to(device)
        dones = torch.BoolTensor([e.done for e in experiences if e is not None]).to(device)
  
        return (states, actions, rewards, next_states, dones)