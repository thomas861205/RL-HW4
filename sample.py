import gym
import numpy as np
import matplotlib.pyplot as plt
from agent import Tabular_Q_learning, Policy_gradient, Actor_critics
from copy import deepcopy


episodes = 1000
reward_hist = np.zeros((episodes))

env = gym.make('CartPole-v1')
agent = Tabular_Q_learning(env)

for episode in range(episodes):
	if episode == episodes-1:
		render = True
	else:
		render = False

	state = env.reset()
	agent.epsilon *= agent.epsilon_decay
	for step in range(500):
	    if render:
	    	# env.render()
	    	# print(state)
	    	# print(agent.state_coding(state))
	    	pass
	    action = agent.act(state)
	    state_next, reward, done, info = env.step(action) # take a random action
	    if done and step < 499:
	    	reward = -1e5
	    agent.update_Q(state, action, reward, state_next, done)
	    state = state_next

	    if done:
	    	reward_hist[episode] = step
	    	env.close()
	    	break

	if episode % 1 == 0:
		print('Episode {}/{} Score: {} Epsilon:{:.5f}'.format(episode, episodes, step, agent.epsilon))


print('avg. {}'.format(np.mean(reward_hist)))
print('last 100 {}'.format(np.mean(reward_hist[-100:])))
print('peak {}'.format(np.max(reward_hist)))

plt.plot(reward_hist)
plt.show()


