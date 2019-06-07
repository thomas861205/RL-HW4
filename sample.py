import gym
import numpy as np
import matplotlib.pyplot as plt
from agent import Tabular_Q_learning, Policy_gradient, Actor_critics
from copy import deepcopy


episodes = 5000
reward_hist = np.zeros((episodes))

env = gym.make('CartPole-v1')

agent = Tabular_Q_learning(env)
for episode in range(episodes):
	state = env.reset()
	for step in range(1000):
	    # env.render()
	    action = agent.act(state)
	    state_next, reward, done, info = env.step(action) # take a random action
	    agent.update_Q(state, action, reward, state_next, done)
	    state = deepcopy(state_next)

	    if done:
	    	reward_hist[episode] = step
	    	break
	if episode % 100 == 0:
		print('Episode {}/{} Score: {} Epsilon:{:.5f}'.format(episode, episodes, step, agent.epsilon))
env.close()
# print(agent.max_cart_vel, agent.min_cart_vel)
# print(agent.max_pole_vel, agent.min_pole_vel)

print('avg. {}'.format(np.mean(reward_hist)))
print('last 100 {}'.format(np.mean(reward_hist[-100:])))
print('peak {}'.format(np.max(reward_hist)))

plt.plot(reward_hist)
plt.show()


