import cv2
import gymnasium as gym
import numpy as np
from collections import deque
from gymnasium.spaces import Box
from math import log


def add_env_wrappers(env):
    env = SkipFramesEnv(env,skip=16)
    env = ResizeFrame(env)
    env = FrameReshape(env)
    env = FrameStack(env,num_stack=4)
    env = NormalizeFrame(env)
    
    return env





class SkipFramesEnv(gym.Wrapper):
    def __init__(self,env,skip):
        super().__init__(env)
        self.__obs_buffer = deque(maxlen=skip)
        self._skip = skip

    def step (self,action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs,reward,done,truncated,info = self.env.step(action)
            self.__obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self.__obs_buffer),axis=0)
        return max_frame, total_reward,  done,truncated,info
    
    '''
    def reset(self):
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs
    '''


class ResizeFrame(gym.ObservationWrapper):
    def __init__(self,env=None):
        super().__init__(env)
        self._observation_space = gym.spaces.Box(low=0,high=255, shape=(84,84,1),dtype=np.uint8)

    @staticmethod
    def process(frame):
        resized_frame = cv2.resize(frame,(84,84),interpolation=cv2.INTER_AREA)
        processed_frame = np.reshape(resized_frame,[84,84,1])
        processed_frame = np.array(processed_frame).astype(np.float32)
        return processed_frame
    
    def observation(self,obs):
        return ResizeFrame.process(obs)
    
class FrameStack(gym.ObservationWrapper,gym.utils.RecordConstructorArgs):
    def __init__(self,env,num_stack):
        super().__init__(env)
        old_space = self.observation_space
        old_shape = self.observation_space.shape
        new_shape =  (old_shape[0]*num_stack,old_shape[1],old_shape[2])
       
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)
       
        self.observation_space = Box(
            low=  old_space.low.repeat(num_stack, axis=0),
            high=old_space.high.repeat(num_stack, axis=0), 
            shape = new_shape,
            dtype=self.observation_space.dtype
        )
   


    def observation(self,observation):
        
        if len(self.frames)<self.num_stack:
            while len(self.frames) < 4:
                self.frames.append(observation)

        self.frames.append(observation)
        frames = [frame.squeeze(0) for frame in self.frames]
        stacked_frames = np.stack(frames,axis=0)
        #return list(self.frames)
        return stacked_frames

class FrameReshape(gym.ObservationWrapper):
    def __init__(self,env):
        super().__init__(env)
        old_shape = self.observation_space.shape
        new_shape =  (old_shape[-1], old_shape[0], old_shape[1])
        self.observation_space = gym.spaces.Box(
            low = 0.0,
            high=1.0,
            shape = new_shape,
            dtype=np.float32
        )
    def observation(self,observation):
        #observation = np.array(observation).astype(np.float32)/255.0
        return np.moveaxis(observation,2,0)
    
class NormalizeFrame(gym.ObservationWrapper):
    def observation(self,observation):
        return np.array(observation).astype(np.float32) / 255.0

#function to normalize and transform rewards
def transform_reward(reward,dead):
    if dead:
        reward = -log(20,1000)
        '''
    if reward > 10 and reward <100:
        reward = 10
    if reward > 200:
        reward = reward / 10
    '''
    return log(reward, 1000) if reward > 0 else reward
   
  
    return reward

def skip_initial_frames(env):
    for i in range(16):
         next_state,reward,done,truncated,info, = env.step(0)