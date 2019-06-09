import numpy as np
from collections import defaultdict
import keras.layers as layers
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
from keras.initializers import glorot_uniform


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
	def __init__(self, env):
		self.env = env
		self.n_actions = env.action_space.n
		self.n_hidden = 8
		self.gamma = 0.95
		self.dimen = len(env.reset())
		self.lr = 1e-2
		self.build_model()
		self.states = np.empty(0).reshape(0,self.dimen)
		self.actions = np.empty(0).reshape(0,1)
		self.rewards = np.empty(0).reshape(0,1)
		self.discounted_rewards = np.empty(0).reshape(0,1)
		# print(self.model_predict.summary())


	def build_model(self):
	    x = layers.Input(shape=self.env.reset().shape, name="x")
	    adv = layers.Input(shape=[1], name="advantages")
	    h1 = layers.Dense(self.n_hidden, 
	                     activation="relu", 
	                     use_bias=False,
	                     kernel_initializer=glorot_uniform(),
	                     name="hidden_1")(x)
	    out = layers.Dense(self.env.action_space.n, 
	                       activation="softmax", 
	                       kernel_initializer=glorot_uniform(),
	                       use_bias=False,
	                       name="out")(h1)

	    def _loss(y_true, y_pred):
	        log_lik = K.log(y_true * (y_true - y_pred) + (1 - y_true) * (y_true + y_pred))
	        return K.mean(log_lik * adv, keepdims=True)

	    self.model_train = Model(inputs=[x, adv], outputs=out)
	    self.model_train.compile(loss=_loss, optimizer=Adam(self.lr))
	    self.model_predict = Model(inputs=[x], outputs=out)




	def discount_rewards(self, rewards):
	    prev = 0
	    ret = []
	    for reward in rewards:
	        curr = reward + self.gamma * prev
	        ret.append(curr)
	        prev = curr
	    return np.array(list(reversed(ret)))


	def act(self, observation):
	    state = np.reshape(observation, [1, self.dimen])
	    
	    predict = self.model_predict.predict([state])[0]
	    action = np.random.choice(range(self.n_actions),p=predict)
	    self.states = np.vstack([self.states, state])
	    self.actions = np.vstack([self.actions, action])

	    return action


	def train(self, states, actions, discounted_rewards):
	    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / discounted_rewards.std()
	    discounted_rewards = discounted_rewards.squeeze()
	    actions = actions.squeeze().astype(int)
	   
	    actions_train = np.zeros([len(actions), self.n_actions])
	    actions_train[np.arange(len(actions)), actions] = 1
	    
	    loss = self.model_train.train_on_batch([states, discounted_rewards], actions_train)

	    states = np.empty(0).reshape(0,self.dimen)
	    actions = np.empty(0).reshape(0,1)
	    discounted_rewards = np.empty(0).reshape(0,1)
	    return loss


	def test(self, model, num_tests):
	    scores = []    
	    for num_test in range(num_tests):
	        observation = self.env.reset()
	        reward_sum = 0
	        while True:
	            state = np.reshape(observation, [1, self.dimen])
	            predict = model.predict([state])[0]
	            action = np.argmax(predict)
	            observation, reward, done, _ = self.env.step(action)
	            reward_sum += reward
	            if done:
	                break
	        scores.append(reward_sum)
	    self.env.close()
	    return np.mean(scores)


class Actor_critics():
	def __init__(self):
		pass

if __name__ == '__main__':
	pass