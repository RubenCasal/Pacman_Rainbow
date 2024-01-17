import gymnasium as gym

import numpy as np
from raimbow_agent import RaimbowAgent
import matplotlib.pyplot as plt

from utils import  add_env_wrappers


EPISODES = 2000000
ENV_NAME = 'ALE/MsPacman-v5'
EPISODES_TEST = 100

TRAINING_PATH = './results'

class Environment:
    def __init__(self,mode,device):
        if mode=='train':
            self.env = gym.make(ENV_NAME,obs_type="grayscale",frameskip=1,full_action_space=False)
         
        else:
            self.env = gym.make(ENV_NAME,obs_type="grayscale",render_mode='human',frameskip=1,full_action_space=False)

        self.env = add_env_wrappers(self.env)
    
        self.env.reset()

        print("Atari Game: "+ ENV_NAME)
        
        state_size = self.env.observation_space.shape
      
        action_size = self.env.action_space.n
        print("aaaaaa")
        print(state_size)

        if mode.lower()== 'test':
            load_model= True
        else:
            load_model = False
      
       
        self.agent = RaimbowAgent(state_size,action_size,device,load_model)

    def train(self,mode):
        print("Start trainning")
        env = self.env
        agent = self.agent
        scores, episodes,mean_scores = [], [], []
        episode_num = 0
        step_counter = 0
        for e in range(EPISODES):
            done = False
            score = 0
            state = env.reset()[0]
           
            lives = 3
            episode_num += 1
           
            
            #Ploting the training performance
            plt.ion() #interactive mode (dinamic graph)
            fig,ax = plt.subplots()
            ax.set_title("Reward per Episode")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Reward")

            while not done:
                dead=False
               
                while not dead:
                   
                    action = agent.get_action(state)
               
                  
                    next_state,reward,done,truncated,info, = env.step(action)
                    
                 
                  
                    #Normalization of pixels

                  
                    
                    agent.append_sample(state,action,reward,next_state,done)
                    
                    agent.train(step_counter)
                    step_counter +=1
                    
                    state = next_state
                    score += reward
                    
                    
                    dead = info['lives']<lives
                    lives = info['lives']

                    #punish when pacman die
                    reward = reward if not dead else -100

               

                if done:
                    print("The game is over")
                    print("Episode: "+ str(episode_num)+" Final Score: "+ str(score))
                    print("Step Counter: "+ str(step_counter)+ " Epsilon: " + str(agent.epsilon))


                    scores.append(score)
                    episodes.append(e)
                    mean_score_value = np.mean(scores[-100:])
                    mean_scores.append(mean_score_value)
                    ax.clear()
                    ax.plot(episodes,scores)
                    ax.plot(episodes,mean_scores,color='r', linestyle='-', linewidth=2, label='Mean Reward')

                    ax.set_title("Reward per Episode")
                    ax.set_xlabel("Episode")
                    ax.set_ylabel("Reward")
                  
                    plt.savefig("./performance_plots/pacman.png")
                    plt.close()
                    plt.pause(0.001)
                

            
            if (e%50==0) and (mode.lower() != 'test'):
                print("Saving Model")
                agent.model.save_model(TRAINING_PATH+"/pacmanTorch.pth")

    def test(self):
       
        env = self.env
    

    
        agent = self.agent
        
        episode_num = 0
        for e in range(EPISODES):
            done = False
            score = 0
            state = env.reset()[0]
                
            lives = 3
            episode_num += 1
            attempt_number = 0


            while not done:
                dead=False
            
                while not dead:
                    
                    action = agent.get_action(state)
                    
                    next_state,reward,done,truncated,info = env.step(action)
                
                    next_state = next_state
                    
                
                    agent.append_sample(state,action,reward,next_state,done)
                    


                    state = next_state
                    score += reward
                    
                    
                    dead = info['lives']<lives
                    lives = info['lives']

                    #punish when pacman die
                    reward = reward if not dead else -100

                attempt_number +=1
                print("Attempt_number: "+str(attempt_number)+" Score: "+str(score))

                if done:
                    print("The game is over")
                   
                

          
                    
                
          






        

