from keras.layers import Dense, Activation
from keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
import numpy as np


class ReplayBuffer(object):
	def __init__(self, max_size, input_shape, n_actions, discrete = False):
		self.mem_size = max_size
		self.mem_cntr = 0
		self.n_actions = n_actions
		self.discrete = discrete
		self.state_memory = np.zeros((self.mem_size, input_shape))
		self.new_state_memory = np.zeros((self.mem_size, input_shape))
		self.reward_memory = np.zeros(self.mem_size)
		dtype = np.int8 if self.discrete else np.float32
		self.action_memory = np.zeros((self.mem_size, n_actions), dtype = dtype)
		self.terminal_memory = np.zeros(self.mem_size)

	def store_transition(self, state, action, reward, state_, done):
		index = self.mem_cntr % self.mem_size
		self.state_memory[index] = state
		self.new_state_memory[index] = state_
		self.reward_memory[index] = reward
		self.terminal_memory[index] = 1- done
		if self.discrete:
			actions = np.zeros(self.n_actions)
			actions[action] = 1.0
			self.action_memory[index] = actions
		else:
			self.action_memory[index] = actions
		self.mem_cntr += 1

	def sample_batch(self, batch_size):
		max_mem = min(self.mem_size, self.mem_cntr)
		batch = np.random.choice(max_mem, batch_size)
		states = self.state_memory[batch]
		actions = self.action_memory[batch]
		rewards = self.reward_memory[batch]
		states_ = self.new_state_memory[batch]
		terminals = self.terminal_memory[batch]
		return states, actions, rewards, states_, terminals

def build_dqn(lr, n_actions, input_dims, fc1_dims, fc2_dims):
	model = Sequential([Dense(fc1_dims, input_shape=(input_dims, )),
						Activation('relu'),
						Dense(fc2_dims),
                        Activation('relu'),
						Dense(n_actions)])
	model.compile(optimizer = Adam(learning_rate = lr), loss = 'mse')

	return model


class DDQNAgent(object):
	def __init__(self, alpha, gamma, n_actions, epsilon, batch_size,
				input_dims, epsilon_dec = 0.995, epsilon_end = 0.01,
				mem_size=100000, f_name='ddqn_model.h5', replace_target=100):
		self.n_actions = n_actions
		self.action_space = [i for i in range(n_actions)]
		self.gamma = gamma
		self.epsilon = epsilon
		self.epsilon_dec = epsilon_dec
		self.batch_size = batch_size
		self.epsilon_min = epsilon_end
		self.memory = ReplayBuffer(mem_size, input_dims, n_actions, True)
		self.model_file = f_name
		self.replace_target = replace_target
		self.q_eval = build_dqn(alpha, n_actions, input_dims, 128, 128)
		self.q_target = build_dqn(alpha, n_actions, input_dims, 128, 128)

	def remember(self, state, action, reward, state_, done):
		self.memory.store_transition(state, action, reward, state_, done)

	def choose_action(self, state):
		state = [state]
		#state = state[np.newaxis, :]
		rand = np.random.random()
		if rand < self.epsilon:
			action = np.random.choice(self.action_space)
		else:
			actions = self.q_eval.predict(state, verbose=0)
			action = np.argmax(actions)
		return action

	def learn(self):
		if self.memory.mem_cntr > self.batch_size:
			state, action, reward, new_state, done = \
			self.memory.sample_batch(self.batch_size)
			action_values = np.array(self.action_space, dtype=np.int8)
			action_indieces = np.dot(action, action_values)
            #k odhadu hodnoty
			q_next = self.q_target.predict(new_state, verbose=0)
			#k výběru maximální akce
			q_eval = self.q_eval.predict(new_state, verbose=0)
			q_pred = self.q_eval.predict(state, verbose=0)

			max_action = np.argmax(q_eval, axis=1)
			q_target = q_pred

			#batch_index = np.arrange(self.batch_size, np.int32)
			batch_index = np.array([j for j in range(self.batch_size)])
			#měním jenom tu akci, co jsem udělal
			q_target[batch_index, action_indieces] = reward +\
			self.gamma*q_next[batch_index, max_action.astype(int)]*done

			_  = self.q_eval.fit(state, q_target, verbose=0)

			self.epsilon = self.epsilon*self.epsilon_dec if self.epsilon > \
			self.epsilon_min else self.epsilon_min
			if self.memory.mem_cntr%self.replace_target == 0:
				self.update_network_parameters()

	def update_network_parameters(self):
		self.save_model()
		self.q_target = load_model(self.model_file)

	def save_model(self):
		self.q_eval.save(self.model_file)

	def loading_model(self):
		self.q_eval = load_model(self.model_file)
		self.q_target = load_model(self.model_file)

		self.epsilon = self.epsilon_min