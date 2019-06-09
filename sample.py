import gym
import numpy as np
import matplotlib.pyplot as plt
from agent import Tabular_Q_learning, Policy_gradient, Actor_critic
from copy import deepcopy


def agent_Q():
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



def agent_pg():
	episodes = 3000
	batch_size = 50
	episode = 0
	losses = []
	reward_hist = []
	test_score = []
	env = gym.make('CartPole-v1')
	agent = Policy_gradient(env)

	while episode < episodes:
		observation = agent.env.reset()
		for step in range(500):

		    action = agent.act(observation)
		    observation, reward, done, _ = agent.env.step(action)
		    if done and step < 499:
		        reward = 1
		    agent.rewards = np.vstack([agent.rewards, reward])
		    
		    if done:
		        reward_hist.append(step)

		        discounted_rewards_episode = agent.discount_rewards(agent.rewards)       
		        agent.discounted_rewards = np.vstack([agent.discounted_rewards, discounted_rewards_episode])
		        agent.rewards = np.empty(0).reshape(0,1)

		        if (episode + 1) % batch_size == 0:
		        	loss = agent.train(agent.states, agent.actions, agent.discounted_rewards)
		        	losses.append(loss)

		        
		        every = 100
		        if (episode + 1) % every == 0:
		            score = agent.test(10)
		            test_score.append(score)
		            print("Avg steps for episode {}/{}: {:0.2f} Test Score: {:0.2f}".format(
		                (episode + 1), episodes, sum(reward_hist[-every:])/every, test_score[-1]))
        
		        episode += 1
		        break

	plt.plot(test_score)
	plt.show()


if __name__ == '__main__':
	# agent_Q()
	agent_pg()
