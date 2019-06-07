import numpy as np
from collections import defaultdict

# Observation: 
#     Type: Box(4)
#     Num	Observation            Min         Max
#     0	Cart Position             -4.8            4.8 -> terminate when |cart position| > 2.4
#     1	Cart Velocity             -Inf            Inf
#     2	Pole Angle    -24 deg = -0.42 radian  24 deg = -0.42 radian -> terminate when |pole angle| > 12deg
#     3	Pole Velocity At Tip      -Inf            Inf


class Tabular_Q_learning():
	def __init__(self, env):
		self.env = env
		self.alpha = 0.3
		self.gamma = 0.95
		self.epsilon = 0.5
		self.epsilon_decay = 0.99
		self.epsilon_min = 0.0


		# self.cart_pos_bin = np.linspace(-2.4, 2.4, num=5)[1:-1]
		self.cart_pos_bin = np.array([-2.0, 2.0])
		# self.cart_vel_bin = np.linspace(-0.5, 0.5, num=5)[1:-1]
		self.cart_vel_bin = np.array([-0.5, 0.5])
		self.pole_ang_bin = np.linspace(-0.1, -0.1, num=11)[1:-1]
		self.pole_vel_bin = np.linspace(-0.5, 0.5, num=5)[1:-1]

		self.Q = defaultdict(lambda: 0*np.ones(env.action_space.n))

	def state_coding(self, state):
		cart_pos = np.digitize(state[0], self.cart_pos_bin)
		cart_vel = np.digitize(state[1], self.cart_vel_bin)
		pole_ang = np.digitize(state[2], self.pole_ang_bin)
		pole_vel = np.digitize(state[3], self.pole_vel_bin)
		
		# ret = (np.asscalar(cart_pos), np.asscalar(cart_vel), np.asscalar(pole_ang), np.asscalar(pole_vel))
		ret = (np.asscalar(cart_pos), 0, np.asscalar(pole_ang), np.asscalar(pole_vel))
		# ret = (0, 0, np.asscalar(pole_ang), np.asscalar(pole_vel))
		return ret


	def update_Q(self, state, action, reward, state_next, done):
		coded_state = self.state_coding(state)
		coded_state_next = self.state_coding(state_next)

		if done:
			target = reward
		else:
			target = reward + self.gamma * max(self.Q[coded_state_next])
		self.Q[coded_state][action] += self.alpha * (target - self.Q[coded_state][action])


	def act(self, state):
		coded_state = self.state_coding(state)

		if np.random.rand() < self.epsilon:
			if self.epsilon > self.epsilon_min:
				self.epsilon *= self.epsilon_decay 
			action = self.env.action_space.sample()
		else:
			action = np.argmax(self.Q[coded_state])
		return action


class Policy_gradient():
	def __init__(self):
		pass


class Actor_critics():
	def __init__(self):
		pass