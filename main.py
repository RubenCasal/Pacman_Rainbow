from environment import Environment
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Modelos para probar de las distintas estrategias
#pacmanTorch_all_same_reward.pth | Todas las acciones tienen la misma recompensa
#pacmanTorch_no_rewards_eat_ghosts.pth | No obtiene recompensa por comer fantasmas
#pacmanTorch_standar_rewards.pth | Con las recompensas que ofrece inicialmente el entorno de gymnasium
#pacmanTorch_rewards_keep_living.pth | Obtiene recompensas periodicamente por sobrevivir cuando alcanza un número determinado de pasos
#pacmanTorch_best_model_log.pth | Mejor modelo entrenado con recompensas normalizadas


MODEL_PATH = './experimentation_models/pacmanTorch_best_model_log.pth' # Aquí modelo que se quiera probar para el testing
SAVE_MODEL_PATH = './experimentation_models/pacmanTorch.pth' #Ruta donde se va a guardar el modelo
MODE = 'test' # test | train
GRAPH_PATH = './experimentation/pacman.png'
EPISODES_TRAIN = 12001
LEARNING_RATE = 0.0001
OPTIMIZER = 'adam' # 'adam' | 'sgd' | 'rmsprop' | 'adamw'
BUFFER_SIZE = 40000

environment = Environment(mode=MODE,device=device,model_path=MODEL_PATH,graph_path = GRAPH_PATH,save_model_path=SAVE_MODEL_PATH, EPISODES_TRAIN = EPISODES_TRAIN, learning_rate=LEARNING_RATE, optimizer= OPTIMIZER,buffer_size=BUFFER_SIZE)
environment.train(mode=MODE) 
     