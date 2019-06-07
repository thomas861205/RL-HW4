import gym
import numpy as np
import matplotlib.pyplot as plt
from agent import Tabular_Q_learning, Policy_gradient, Actor_critics
from copy import deepcopy


episodes = 10000
reward_hist = np.zeros((episodes))

env = gym.make('CartPole-v1')
agent = Tabular_Q_learning(env)

for episode in range(episodes):
	if episode % 1000 == 0:
		render = True
	else:
		render = False

	state = env.reset()
	for step in range(1000):
	    if render:
	    	env.render()
	    	# print(state)
	    	# print(agent.state_coding(state))
	    action = agent.act(state)
	    state_next, reward, done, info = env.step(action) # take a random action
	    agent.update_Q(state, action, reward, state_next, done)
	    state = state_next

	    if done:
	    	reward_hist[episode] = step
	    	env.close()
	    	break

	if episode % 100 == 0:
		print('Episode {}/{} Score: {} Epsilon:{:.5f}'.format(episode, episodes, step, agent.epsilon))
# print(agent.max_cart_vel, agent.min_cart_vel)
# print(agent.max_pole_vel, agent.min_pole_vel)

print('avg. {}'.format(np.mean(reward_hist)))
print('last 100 {}'.format(np.mean(reward_hist[-100:])))
print('peak {}'.format(np.max(reward_hist)))

plt.plot(reward_hist)
plt.show()


