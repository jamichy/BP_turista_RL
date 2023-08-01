from environment import GridWorld
from utils import plotLearning
import numpy as np
import time
import random

def maxAction(Q, state):
	values = np.array([Q[state,a] for a in env.possibleActions])
	action = np.argmax(values)
	return action


if __name__=='__main__':
	start_time = time.process_time()
	print("Let's play")
	m = 16
	n = 16
	env = GridWorld(m, n, 'terrain_1.csv', 0)
	n_episodes = 751
	epsilon = 1.0
	gamma = 1.0
	alpha = 0.3


	Q_history_score = []
	eps_history = []
	Q = {}
	for state in env.stateSpacePlus:
		for action in env.possibleActions:
			Q[state, action] = 0

	for i in range(n_episodes):
		done  = False
		score = 0
		observation = env.reset()
		while not done:
			rand = np.random.random()
			if rand < epsilon:
				action = random.choice(env.possibleActions)
			else:
				action = maxAction(Q, observation)
			observation_, reward, done = env.step(action)
			score += reward

			#learning_phase
			max_next_action = maxAction(Q, observation_)
			Q[observation, action] = Q[observation, action] + alpha*(reward + gamma* Q[observation_, max_next_action]- Q[observation, action])

			observation = observation_
		Q_history_score.append(score)


            
		#Image_folder/
		env.workingImage.save('Image_folder/hra-{0},skore-{1}.png'.format(i+1, score))	
		eps_history.append(epsilon)
		avg_score = np.mean(Q_history_score[max(0, i-10): i+1])
		print('episode ', i, 'score %.2f' %score, 'average_score %.2f' %avg_score, 'epsilon %.2f' %epsilon)
		if i == (n_episodes - 2):
			epsilon = 0.0
		else:
			epsilon -= 1/(n_episodes-1)
	elapsed_time = time.process_time() - start_time
	print("Elapsed time: ", elapsed_time)
	filename = 'Q_learning.png'
	plotLearning([x+1 for x in range(n_episodes)], Q_history_score, eps_history, filename)
