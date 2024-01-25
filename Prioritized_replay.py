from collections import deque
import random
import numpy as np
import torch
import collections

Experience = collections.namedtuple("Experience",field_names=['state','action','reward','done','next_state','priority'])
class Prioritized_Replay:
    def __init__(self,capacity):
        self.capacity = capacity
        #self.buffer = deque(maxlen=capacity)
        self.len_buffer = 0
        self.buffer = np.empty(self.capacity, dtype=[("priority", np.float32), ("experience", Experience)])
       
    #add experience 
    def add_experience(self,state,action,reward,done,next_state):
        priority = 1.0 if self.size()==0 else self.buffer["priority"].max()
        exp = Experience(state,action,reward,done,next_state,priority)
        if self.size() == self.capacity:
            if priority>self.buffer["priority"].min():
                    idx = self.buffer["priority"].argmin()
                    self.buffer[idx] = (priority, exp)
            else:
                    pass
        else:
                self.buffer[self.size()] = (priority, exp)  
                self.len_buffer +=1     


        

    def size(self):
        return self.len_buffer
    
    def clear(self):
        self.buffer.clear()

    def sample(self,batch_size,device):

        batch_size = min(batch_size,self.size())
       
        #priorities = self.buffer[:self.size()]["priority"]
        priorities = self.buffer[:self.size()]["priority"]
      
        n_priorities = priorities/ priorities.sum()
        idxs = np.random.choice(np.arange(priorities.size),
                                         size=batch_size,
                                         replace=True,
                                         p=n_priorities)

        experiences = self.buffer["experience"][idxs]  

       

        states = torch.tensor(np.array([e.state for e in experiences if e is not None])).to(device)
        next_states = torch.tensor(np.array([e.next_state for e in experiences if e is not None])).to(device)
        actions = torch.tensor(np.array([e.action for e in experiences if e is not None],dtype=np.int64)).to(device)
        rewards=  torch.tensor([e.reward for e in experiences if e is not None]).to(device)
        dones = torch.BoolTensor([e.done for e in experiences if e is not None]).to(device)
  
        return (idxs,states, actions, rewards, next_states, dones)
    
    def update_priorityies(self,idxs,priorities):
         
       
         self.buffer["priority"][idxs] = priorities