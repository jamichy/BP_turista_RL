# BP_turista_RL
## Introduction
This repository contains the code for a bachelor thesis on Automatic stock trading based on fundamental factors and reinforcement learning. For this bachelor thesis a demonstration problem is designed. It is accompanied by the code, which is divided into folders. Each contains separately functional code to illustrate different RL algorithms for example Monte Carlo, Q-learning, Double Q-learning, Sarsa, DQN. 

## Demonstration task
Consider a problem where the agent is in the terrain and its goal is to discover a path from a starting position to a final position that consumes the least amount of energy. In doing so, the agent does not know where the final position is and must therefore explore the map thoroughly to find the final position. The terrain map is represented by square boxes with the corresponding altitude. The agent can move up, down, right, left and diagonally one square at a time. If he takes an action that would take him off the map, he remains on the the same square and receives a -10 reward. He receives a reward of 5 for an action that results in a final state, regardless of the height of the square from which he took the action. Any action that does not lead to the final position costs the agent energy - the agent takes a penalty from the environment. This penalty is higher the greater the difference in heights of the current and next square where the action is taken, and also if the action took a diagonal path instead of moving to one of the cardinal directions. We chose to have the agent start in the bottom left corner and the final position be in the top right corner. There is also an optional height factor L, which expresses the ratio of the energy consumed to overcome a height separation of length 1 versus the energy required to move in a plane of the same length.

The base map size is default set to 16 × 16 and the height factor L is set to 2.

The task environment is programmed in the file environment.py. It includes initialization of class GridWorld and definition of reset and step methods. The step method takes the agent's action as input and calculates the subsequent state and reward. On output, it returns the new state, the reward, and a boolean value of whether the new state is final. In the reset method, the agent's state is set to initial and this state is returned to the agent. The method is called in the main program before the start of each episode.

In the initialization of the GridWorld class we read the terrain map from the csv file terrain_1.csv. The loaded terrain looks like this.
![terrain](https://github.com/jamichy/BP_turista_RL/assets/112120789/787bf958-22d1-4bcd-a821-f12f2c6080cb)


For each episode, a representation of the agent's route is saved in the Images folder. 

## Algorithms
Let's describe used methods. As mentioned earlier, each algorithm is programmed in the main file of its own folder. The first step in the main program is to import https://github.com/jamichy/BP_turista_RL/blob/main/MC/environment.py, other usefull libraries and initialize the environment by the following code.
```python
from environment import GridWorld


#width adjustment 
m = 16
#height adjustment
n = 16
#initialization of the environment
env = GridWorld(m, n, 'terrain_1.csv', 2)
#setting the number of episodes
n_episodes = 10001
```

### Monte Carlo


#### Usage

### Sarsa


### Q-learning


### Double Q-learning



### DQN
...
```python
import foobar

# returns 'words'
foobar.pluralize('word')

# returns 'geese'
foobar.pluralize('goose')

# returns 'phenomenon'
foobar.singularize('phenomena')
```
