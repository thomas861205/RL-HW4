import gym
import numpy as np
from agent import Tabular_Q_learning, Policy_gradient, Actor_critics

episodes = 3000
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
	    	print('Score: {}'.format(step))
	    	break
	env.close()


