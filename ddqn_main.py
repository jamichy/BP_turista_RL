from environment import GridWorld
from ddqn_model import DDQNAgent
from utils import plotLearning
import numpy as np
import time

if __name__=='__main__':
	start_time = time.process_time()
	print("Let's play")
	env = GridWorld(16, 16, 'terrain_1.csv', 2)
	n_episodes = 1
	
	#value that describes if we want to load model and donť trained it
	loading_bool = True

	ddqn_history = []
	eps_history = []
	ddqn_agent= DDQNAgent(alpha = 0.01, gamma = 0.99, n_actions=8, epsilon=1.0,
				batch_size=64, input_dims=2)
	if loading_bool:
		ddqn_agent.loading_model()
		ddqn_agent.epsilon = 0.0
		ddqn_agent.epsilon_dec = 1.00
	 
	for i in range(n_episodes):
		done  = False
		score = 0
		ddqn_agent.epsilon_min = 0.0
		observation = env.reset()
		if not loading_bool:
			if i == 0:
				ddqn_agent.epsilon_dec = 1.00
			elif i<101:
				ddqn_agent.epsilon -= 0.01
			else:		
				ddqn_agent.epsilon -= 0.001
                    

		while not done:
			action = ddqn_agent.choose_action(observation)
			observation_, reward, done = env.step(action)
			ddqn_agent.remember(observation, action, reward, observation_, done)
			score += reward
			observation = observation_
			ddqn_agent.learn()
            
		#Image_folder/
		env.workingImage.save('Image_folder/hra-{0},skore-{1}.png'.format(i+1, score))	
		eps_history.append(ddqn_agent.epsilon)
		ddqn_history.append(score)
		avg_score = np.mean(ddqn_history[max(0, i-10): i+1])
		print('episode ', i, 'score %.2f' %score, 'average_score %.2f' %avg_score, 'epsilon %.2f' %ddqn_agent.epsilon)
	elapsed_time = time.process_time() - start_time
	print("Elapsed time: ", elapsed_time)
	filename = 'ddqn.png'
	plotLearning([x+1 for x in range(n_episodes)], ddqn_history, eps_history, filename)
