import numpy as np
from ple import PLE  # our environment
from ple.games.catcher import Catcher
from ple.games.flappybird import FlappyBird
from ple.games.raycastmaze import RaycastMaze

from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.models import load_model
from six.moves import cPickle

from example_support import ExampleAgent, ReplayMemory, loop_play_forever, loadedModel


class Agent(ExampleAgent):
    """
        Our agent takes 1D inputs which are flattened.
        We define a full connected model below.
    """

    def __init__(self, *args, **kwargs):
        ExampleAgent.__init__(self, *args, **kwargs)

        self.state_dim = self.env.getGameStateDims()
        self.state_shape = np.prod((num_frames,) + self.state_dim)
        self.input_shape = (batch_size, self.state_shape)

    def build_model(self):
        model = Sequential()
        model.add(Dense(input_dim=self.state_shape, output_dim=256, activation="relu", init="he_uniform"))
        model.add(Dense(512, activation="relu", init="he_uniform"))
        model.add(Dense(self.num_actions, activation="linear", init="he_uniform"))

        model.compile(loss=self.q_loss, optimizer=SGD(lr=self.lr))
        self.model = model

    def load_model(self,fileName):
        self.model = load_model(fileName, custom_objects={'q_loss': self.q_loss})


def nv_state_preprocessor(state):
    """
        This preprocesses our state from PLE. We rescale the values to be between
        0,1 and -1,1.
    """
    # taken by inspection of source code. Better way is on its way!
    catcherMax = np.array([128.0, 20.0, 128.0, 128.0])  
    
    flappyMax = np.array([390, 10, 309, 192, 292, 453, 192])

    rayMax = np.array([10,10,1,1,1,10,1])

    state = np.array([list(state.values())])/catcherMax    
          
    #state = np.array([state.values()])/ max_values

    return state.flatten()

if __name__ == "__main__":

    # this takes about 15 epochs to converge to something that performs decently.
    # feel free to play with the parameters below.

    # training parameters
    num_epochs = 30
    num_steps_train = 15000  # steps per epoch of training
    num_steps_ep = 200
    
    num_steps_test = 3000
    update_frequency = 4  # step frequency of model training/updates

    # agent settings
    batch_size = 32
    num_frames = 4  # number of frames in a 'state'
    frame_skip = 2
    # percentage of time we perform a random action, help exploration.
    epsilon = 0.5
    epsilon_steps = 3000  # decay steps
    epsilon_min = 0.1
    lr = 0.01
    discount = 0.95  # discount factor
    rng = np.random.RandomState(24)
    

    # memory settings
    max_memory_size = 100000
    min_memory_size = 1000  # number needed before model training starts

    epsilon_rate = (epsilon - epsilon_min) / epsilon_steps

    
    rewardsVals = {
                "positive": 1.0,
                "negative": -0.01,
                "tick": -0.0,
                "loss": -5.0,
                "win": 5.0
            }

    # PLE takes our game and the state_preprocessor. It will process the state
    # for our agent.
    game = Catcher(128,128)

    #game = FlappyBird()
    #game = RaycastMaze()


    env = PLE(game, fps=60, state_preprocessor=nv_state_preprocessor,reward_values=rewardsVals)

    agent = Agent(env, batch_size, num_frames, frame_skip, lr, discount, rng, optimizer="sgd_nesterov")    

    memory = ReplayMemory(max_memory_size, min_memory_size)

    env.init()

    agent.build_model()

    #PreTrained Model FileNames
    flappyModel = 'flappybird.h5'
    catcherModel = 'catcher.h5'
    rayMazeModel = 'RayMaze.h5'

    #Model save filename
    saveName = rayMazeModel

    

    agent.load_model('catcher-MASTER.h5')

    loadedModel(env,agent)
   
    for epoch in range(1, num_epochs + 1):
        steps, num_episodes = 0, 0
        losses, rewards = [], []
        env.display_screen = True
        stepsPerEp = num_steps_ep
        # training loop
        while steps < num_steps_train:
            episode_reward = 0.0
            agent.start_episode()
            
            
            while env.game_over() == False and steps < num_steps_train: # and steps < stepsPerEp:
                
                state = env.getGameState()
                
                reward, action = agent.act(state, epsilon=epsilon)                
                memory.add([state, action, reward, env.game_over()])                
                if steps % update_frequency == 0:
                    loss = memory.train_agent_batch(agent)

                    if loss is not None:  
                        losses.append(loss)     
                        epsilon = max(epsilon_min, epsilon)

                episode_reward += reward
                steps += 1
                
                
            if num_episodes % 5 == 0:
                print ("Episode {:01d}: Reward {:0.1f}".format(num_episodes, episode_reward))
            
            rewards.append(episode_reward)            
            num_episodes += 1
            stepsPerEp += num_steps_ep
            agent.end_episode()

        print ("\nTrain Epoch {:02d}: Epsilon {:0.4f} | Avg. Loss {:0.3f} | Avg. Reward {:0.3f}".format(epoch, epsilon, np.mean(losses), np.sum(rewards) / num_episodes))

        steps, num_episodes = 0, 0
        losses, rewards = [], []

        # display the screen
        env.display_screen = True
        agent.model.save(saveName)     
      
    print ("\nTraining complete. Will loop forever playing!")  
    
    agent.model.save(saveName)
   
    loop_play_forever(env, agent)
    


    