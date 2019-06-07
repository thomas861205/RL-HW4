import numpy as np
from collections import defaultdict

class Tabular_Q_learning():
	def __init__(self, env):
		self.env = env
		self.alpha = 0.1
		self.gamma = 0.98
		self.epsilon = 0.05
		self.epsilon_decay = 0.999
		self.Q = defaultdict(lambda: np.zeros(env.action_space.n))

	def state_coding(self, state):
		cart_pos_bin = np.linspace(-4.8, 4.8, num=2)
		cart_vel_bin = np.linspace(-1, 1, num=2)
		pole_ang_bin = np.linspace(-0.41, 0.41, num=7)
		pole_vel_bin = np.linspace(-0.87, 0.87, num=4)

		cart_pos = np.digitize(state[0], cart_pos_bin)
		cart_vel = np.digitize(state[1], cart_vel_bin)
		pole_ang = np.digitize(state[2], pole_ang_bin)
		pole_vel = np.digitize(state[3], pole_vel_bin)
		
		ret = (np.asscalar(cart_pos), np.asscalar(cart_vel), np.asscalar(pole_ang), np.asscalar(pole_vel))
		print(state)
		print(ret)
		return ret

	def update_Q(self, state, action, reward, state_next, done):
		state = self.state_coding(state)
		state_next = self.state_coding(state_next)
		if done:
			target = reward
		else:
			target = reward + self.gamma * max(self.Q[state_next])
		self.Q[state][action] = self.Q[state][action] + self.alpha * (target - self.Q[state][action])

	def act(self, state):
		state = self.state_coding(state)
		self.epsilon *= self.epsilon_decay 
		if np.random.rand() < self.epsilon:
			action = self.env.action_space.sample()
		else:
			action = np.argmax(self.Q[state])
		return action


class Policy_gradient():
	def __init__(self):
		pass


class Actor_critics():
	def __init__(self):
		pass