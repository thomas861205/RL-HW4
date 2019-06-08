import numpy as np
from collections import defaultdict

# Observation: 
#     Type: Box(4)
#     Num	Observation            Min         Max
#     0	Cart Position             -4.8            4.8 -> terminate when |cart position| > 2.4
#     1	Cart Velocity             -Inf            Inf
#     2	Pole Angle    -24 deg = -0.42 radian  24 deg = -0.42 radian -> terminate when |pole angle| > 12deg
#     3	Pole Velocity At Tip      -Inf            Inf
# or reach timestep 500

class Tabular_Q_learning():
	def __init__(self, env):
		self.env = env
		self.alpha = 0.1
		self.gamma = 1
		self.epsilon = 0.5
		self.epsilon_decay = 0.99
		self.epsilon_min = 0.0


		self.cart_pos_bin = np.linspace(-2.4, 2.4, num=6)[1:-1]
		self.cart_vel_bin = np.linspace(-3, 3, num=4)[1:-1]
		self.pole_ang_bin = np.linspace(-0.21, 0.21, num=8)[1:-1]
		self.pole_vel_bin = np.linspace(-2.0, 2.0, num=6)[1:-1]
		self.Q = defaultdict(lambda: np.zeros(env.action_space.n))

	def state_coding(self, state):
		cart_pos = np.digitize([state[0]], self.cart_pos_bin)[0]
		cart_vel = np.digitize([state[1]], self.cart_vel_bin)[0]
		pole_ang = np.digitize([state[2]], self.pole_ang_bin)[0]
		pole_vel = np.digitize([state[3]], self.pole_vel_bin)[0]
		
		
		return (cart_pos, cart_vel, pole_ang, pole_vel)


	def update_Q(self, state, action, reward, state_next, done):
		coded_state = self.state_coding(state)
		coded_state_next = self.state_coding(state_next)

		target = reward + self.gamma * max(self.Q[coded_state_next])
		self.Q[coded_state][action] += self.alpha * (target - self.Q[coded_state][action])


	def act(self, state):
		coded_state = self.state_coding(state)

		if np.random.uniform(0, 1) < self.epsilon:
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