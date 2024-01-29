import random
import numpy as np
from SumTree import SumTree
import collections
import torch

Experience = collections.namedtuple("Experience",field_names=['state','action','reward','done','next_state','priority'])
class Prioritized_Buffer_Replay:
    def __init__(self,capacity):
        self.capacity = capacity
        #self.buffer = deque(maxlen=capacity)
        self.len_buffer = 0
        self.buffer = np.empty(self.capacity, dtype=[("priority", np.float32), ("experience", Experience)])
        self.alpha=0.6
        self.beta = 0.4
        self.e = 0.001
        self.beta_increment_per_sampling = 0.001
        self.tree = SumTree(capacity)
        self.batch = 128
        

    def size(self):
        return self.tree.n_entries

    def _get_priority(self, error):
        if isinstance(error, torch.Tensor):
            error = error.detach().cpu().numpy()
        return (np.abs(error) + self.e) ** self.alpha 

    def add(self, error, exp):

        p = self._get_priority(error)
        self.tree.add(p, exp)

    def sample(self):
        batch = []
        idxs = []
        segment = self.tree.total() / self.batch
        priorities = []
        device = 'cuda:0'
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(self.batch):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()


        states = torch.tensor(np.array([e.state for e in batch if e is not None])).to(device)
        next_states = torch.tensor(np.array([e.next_state for e in batch if e is not None])).to(device)
        actions = torch.tensor(np.array([e.action for e in batch if e is not None],dtype=np.int64)).to(device)
        rewards=  torch.tensor([e.reward for e in batch if e is not None]).to(device)
        dones = torch.BoolTensor([e.done for e in batch if e is not None]).to(device)

        return idxs,states,actions,rewards,next_states,dones,is_weight
    
    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)