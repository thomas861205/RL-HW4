import numpy as np
import keras.backend as K
import keras.losses
import keras.layers as layers
from keras.models import Model
from keras.optimizers import Adam
from keras.initializers import glorot_uniform
from collections import defaultdict
import matplotlib.pyplot as plt


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
		self.n_hidden = 128
		self.gamma = 0.99
		self.dimen = len(env.reset())
		self.lr = 0.01
		self.build_model()
		self.states = np.empty(0).reshape(0,self.dimen)
		self.actions = np.empty(0).reshape(0,1)
		self.rewards = np.empty(0).reshape(0,1)
		self.discounted_rewards = np.empty(0).reshape(0,1)


	def build_model(self):
	    x = layers.Input(shape=self.env.reset().shape, name="x")
	    adv = layers.Input(shape=[1], name="advantages")
	    h1 = layers.Dense(self.n_hidden, 
	                     activation="relu", 
	                     # kernel_initializer=glorot_uniform(),
	                     kernel_initializer='ones',
	                     use_bias=False,
	                     )(x)

	    # d1 = layers.Dropout(0.6, input_shape=(self.n_hidden,))(h1)

	    out = layers.Dense(self.env.action_space.n, 
	                       activation="softmax", 
	                       # kernel_initializer=glorot_uniform(),
	                       kernel_initializer='ones',
	                       use_bias=False)(h1)

	    def _loss(y_true, y_pred):
	        log_lik = -y_true * K.log(y_pred + 1e-15)
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


	def test(self, num_tests):
	    scores = []    
	    for num_test in range(num_tests):
	        observation = self.env.reset()
	        reward_sum = 0
	        while True:
	            state = np.reshape(observation, [1, self.dimen])
	            predict = self.model_predict.predict([state])[0]
	            action = np.argmax(predict)
	            # action = np.random.choice(range(self.n_actions),p=predict)
	            observation, reward, done, _ = self.env.step(action)
	            reward_sum += reward
	            if done:
	                break
	        scores.append(reward_sum)
	    self.env.close()
	    return np.mean(scores)


class Actor_critic():
    def __init__(self, env):
        self.env = env
        self.gamma = 0.99
        self.actor_lr = 0.001
        self.critic_lr = 0.001 #0.01
        self.build_actor()
        self.build_critic()


    def build_actor(self):
        inputs = layers.Input(shape=(4,))
        h1 = layers.Dense(32, activation='relu', kernel_initializer='ones')(inputs)
        # h2 = layers.Dense(20, activation='relu', kernel_initializer='ones')(h1)
        d1 = layers.Dropout(0.6, input_shape=(32,))(h1)
        out = layers.Dense(1, activation='sigmoid', kernel_initializer='ones')(d1)
        self.actor = Model(inputs=inputs, outputs=out)

        def _actor_loss(y_true, y_pred):
            action_pred = y_pred
            action_true, td_error = y_true[:, 0], y_true[:, 1]
            action_true = K.reshape(action_true, (-1, 1))
            loss = K.binary_crossentropy(action_true, action_pred)
            return loss * K.flatten(td_error)

        self.actor.compile(loss=_actor_loss, optimizer=Adam(lr=self.actor_lr))


    def build_critic(self):
        inputs = layers.Input(shape=(4,))
        h1 = layers.Dense(16, activation='relu')(inputs)
        h2 = layers.Dense(16, activation='relu')(h1)
        out = layers.Dense(1, activation='linear')(h2)
        self.critic = Model(inputs=inputs, outputs=out)
        self.critic.compile(loss='mse', optimizer=Adam(lr=self.critic_lr))


    def discount_reward(self, next_states, reward, done):
        q = self.critic.predict(next_states)[0][0]
        target = reward
        if not done:
            target = reward + self.gamma * q
        
        return target


    def act(self, state):
        prob = self.actor.predict(state)[0][0]
        action = np.random.choice(np.array(range(2)), p=[1 - prob, prob])
        return action


    def train(self, state, action, reward, state_next, done):
        target = self.discount_reward(state_next, reward, done)
        y = np.array([target])

        td_error = target - self.critic.predict(state)[0][0]
        loss1 = self.critic.train_on_batch(state, y)

        y = np.array([[action, td_error]])
        loss2 = self.actor.train_on_batch(state, y)
        return loss1, loss2

if __name__ == '__main__':
	pass