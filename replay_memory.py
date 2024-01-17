from collections import deque
import random
import numpy as np

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


  

   

    def sample(self,batch_size):
        
        indices = np.random.choice(len(self.buffer),batch_size,replace=False)
        states,actions,rewards,dones,next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states),np.array(actions),np.array(rewards),np.array(dones),np.array(next_states)

    