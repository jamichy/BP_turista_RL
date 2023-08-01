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
	n_episodes = 1501
	epsilon = 1.0
	gamma = 1.0
	alpha = 0.3


	DQ_history_score = []
	eps_history = []
	Q_1 = {}
	Q_2 = {}
	for state in env.stateSpacePlus:
		for action in env.possibleActions:
			Q_1[state, action] = 0
			Q_2[state, action] = 0

	for i in range(n_episodes):
		done  = False
		score = 0
		observation = env.reset()
		while not done:
			rand = np.random.random()
			if rand < epsilon:
				action = random.choice(env.possibleActions)
			else:
				action = maxAction(Q_1, observation)
			observation_, reward, done = env.step(action)
			score += reward

			#learning_phase
			rand = np.random.random()
			if rand <= 0.5:
				max_next_action = maxAction(Q_1, observation_)
				Q_1[observation, action] = Q_1[observation, action] + alpha*(reward + gamma* Q_2[observation_, max_next_action]- Q_1[observation, action])
			else:
				max_next_action = maxAction(Q_2, observation_)
				Q_2[observation, action] = Q_2[observation, action] + alpha*(reward + gamma* Q_1[observation_, max_next_action]- Q_2[observation, action])

			observation = observation_
		DQ_history_score.append(score)


            
		#Image_folder/
		env.workingImage.save('Image_folder/hra-{0},skore-{1}.png'.format(i+1, score))	
		eps_history.append(epsilon)
		avg_score = np.mean(DQ_history_score[max(0, i-10): i+1])
		print('episode ', i, 'score %.2f' %score, 'average_score %.2f' %avg_score, 'epsilon %.2f' %epsilon)
		if i == (n_episodes-2):
			epsilon = 0.0
		else:
			epsilon -= 1/(n_episodes-1)
	elapsed_time = time.process_time() - start_time
	print("Elapsed time: ", elapsed_time)
	filename = 'DQ_learning.png'
	plotLearning([x+1 for x in range(n_episodes)], DQ_history_score, eps_history, filename)
