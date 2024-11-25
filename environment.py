import gymnasium as gym

import numpy as np
from raimbow_agent import RaimbowAgent
import matplotlib.pyplot as plt

from utils import  add_env_wrappers, skip_initial_frames, transform_reward


EPISODES = 4501
ENV_NAME = 'ALE/MsPacman-v5'
EPISODES_TEST = 3

TRAINING_PATH = './experimentation_models'

class Environment:
    def __init__(self,mode,device,model_path,graph_path,save_model_path,EPISODES_TRAIN,learning_rate,optimizer,buffer_size):
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer
        self.buffer_size = buffer_size
        if mode=='train':
            self.env = gym.make(ENV_NAME,obs_type="grayscale",frameskip=1,full_action_space=False)
            self.load_model = False
            self.model_path = None
            
        else:
            self.env = gym.make(ENV_NAME,obs_type="grayscale",render_mode='human',frameskip=1,full_action_space=False)
            self.load_model= True
            self.model_path = model_path

        self.episodes_train = EPISODES_TRAIN
        #Add wrappers
        self.env = add_env_wrappers(self.env)
        self.graph_path = graph_path
        self.save_model_path = save_model_path
        self.env.reset()

        print("Atari Game: "+ ENV_NAME)
        state_size = self.env.observation_space.shape
        action_size = self.env.action_space.n

        self.agent = RaimbowAgent(state_size=state_size,action_size=action_size,device=device,model_path=self.model_path,load_model=self.load_model,learning_rate=self.learning_rate,optimizer_type=self.optimizer_type,buffer_size=self.buffer_size)

    def train(self,mode):
        print("Start trainning")
        env = self.env
        agent = self.agent
        scores, episodes,mean_scores = [], [], []
        episode_num = 0
       
        step_counter = 0
        max_mean_score =0
        for e in range( self.episodes_train):
            done = False
            score = 0
            state = env.reset()[0]
            episode_count_step =0
            lives = 3
            episode_num += 1

            skip_initial_frames(env)
           
            while not done:
                dead=False

                while not dead:
                    
                    action = agent.get_action(state)
                
                    next_state,reward,done,truncated,info, = env.step(action)
                    if not self.load_model:
                        score += reward
                        dead = info['lives']<lives
                        lives = info['lives']
                        
                        reward = transform_reward(reward,dead, episode_count_step)
                        agent.append_sample(state,action,reward,next_state,done)
                        agent.train(step_counter)
                    episode_count_step +=1
                
                    step_counter +=1
                    state = next_state
                      
                if done:
                    print("Episode: "+ str(episode_num)+" Final Score: "+ str(score))
                    print("Step Counter: "+ str(step_counter)+ " Epsilon: " + str(agent.epsilon))

                    scores.append(score)
                    episodes.append(e)
                    mean_score_value = np.mean(scores[-100:])
                    mean_scores.append(mean_score_value)
                    
                 
            if (e%50==0) and (mode.lower() != 'test'):
                
                if not self.load_model and  (mean_score_value>= max_mean_score) :
                    print("Saving Model")
                    max_mean_score = mean_score_value

                    agent.model.save_model(self.save_model_path)

                #Crear gráfica del rendimiento del modelo
                plt.ion() #modo interactivo (dinamic graph)
                fig,ax = plt.subplots()
                ax.set_title("Reward per Episode")
                ax.set_xlabel("Episode")
                ax.set_ylabel("Reward")
                ax.clear()
                ax.plot(episodes,scores)
                ax.plot(episodes,mean_scores,color='r', linestyle='-', linewidth=2, label='Mean Reward')

                ax.set_title("Reward per Episode")
                ax.set_xlabel("Episode")
                ax.set_ylabel("Reward")
                
                plt.savefig(self.graph_path)
                plt.close()
                    

