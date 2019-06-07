import gym
import numpy as np
import matplotlib.pyplot as plt
from agent import Tabular_Q_learning, Policy_gradient, Actor_critics

episodes = 10000
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
	    state = state_next

	    if done:
	    	reward_hist[episode] = step
	    	break
	if episode % 10 == 0:
		print('Episode {}/{} Score: {}'.format(episode, episodes, step))
	env.close()

print('avg. {}'.format(np.mean(reward_hist)))
print('last 100 {}'.format(np.mean(reward_hist[-100:])))
print('peak {}'.format(np.max(reward_hist)))

plt.plot(reward_hist)
plt.show()


