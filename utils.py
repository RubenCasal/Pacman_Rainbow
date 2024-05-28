import cv2
import gymnasium as gym
import numpy as np
from collections import deque
from gymnasium.spaces import Box
from math import log

# Inicializa el entorno con todos los wrappers
def add_env_wrappers(env):
    env = SkipFramesEnv(env,skip_frames=16)
    env = ReduceActionSpaceEnv(env) 
    env = ResizeFrame(env)
    env = FrameReshape(env)
    env = FrameStack(env,num_stack=4)
    env = NormalizeFrame(env)
    
    return env

# Reduce el número de decisiones a tomar
class SkipFramesEnv(gym.Wrapper):
    def __init__(self,env,skip_frames):
        super().__init__(env)
        # Buffer para almacenar las observaciones de los últimos frames
        self.frame_buffer  = deque(maxlen=skip_frames)
        # Número de frames a omitir
        self.skip_frames  = skip_frames

    def step (self,action):
        accumulated_reward  = 0.0
        done  = None
        # Ejecuta la acción y acumula las recompensas y observaciones por el número de frames especificado
        for _ in range(self.skip_frames ):
            obs,reward,done,truncated,info = self.env.step(action)
            self.frame_buffer.append(obs)
            accumulated_reward += reward
            if done:

                break
        # Toma el máximo valor de los frames almacenados para la observación actual
        max_frame = np.max(np.stack(self.frame_buffer ),axis=0)
        return max_frame, accumulated_reward, done ,truncated,info

# Reduce el tamaño de las obsevaciones
class ResizeFrame(gym.ObservationWrapper):
    def __init__(self,env=None):
        super().__init__(env)
        self._observation_space = gym.spaces.Box(low=0,high=255, shape=(84,84,1),dtype=np.uint8)

    @staticmethod
    def process(frame):
        # Redimensionar la imagen con interpolación bilineal
        resized_frame = cv2.resize(frame,(84,84),interpolation=cv2.INTER_AREA)
        # Reorganizar la forma del array
        reshaped_frame  = resized_frame.reshape((84, 84, 1))
         # Convertir el tipo de dato a float32 para normalización
        normalized_frame  = np.array(reshaped_frame).astype(np.float32)
        return normalized_frame
    
    def observation(self,obs):
        return ResizeFrame.process(obs)
    
# Reduce el número de acciones posibles
class ReduceActionSpaceEnv(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(4)
        self.action_map = {0: 1, 1: 4, 2: 2, 3: 3}

    def action(self, action):
        return self.action_map[action]  

# Apila los frames para dar contexto  
class FrameStack(gym.ObservationWrapper,gym.utils.RecordConstructorArgs):
    def __init__(self,env,num_stack):
        super().__init__(env)
        old_space = self.observation_space
        old_shape = self.observation_space.shape
        new_shape =  (old_shape[0]*num_stack,old_shape[1],old_shape[2])
       
        self.num_stack = num_stack
        # Crea un buffer para almacenar los frames recientes
        self.frames = deque(maxlen=num_stack)
       # Define el nuevo espacio de observación con los límites ajustados para el apilamiento
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

# Ajusta la dimensión para que sea compatible con las redes convolucionales de PyTorch
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
        return np.moveaxis(observation,2,0)
    
# Normaliza los pixeles de las observaciones
class NormalizeFrame(gym.ObservationWrapper):
    def observation(self,observation):
        return np.array(observation).astype(np.float32) / 255.0

#Transforma las recompensas
def transform_reward(reward,dead,episode_step):
    if dead:
        reward = -log(100, 1000)
   
    return log(reward, 1000) if reward > 0 else reward
   

# Salta los frames iniciales del comienzo de un episodio
def skip_initial_frames(env):
    for i in range(16):
         next_state,reward,done,truncated,info, = env.step(0)