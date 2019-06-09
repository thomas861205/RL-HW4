import gym
import numpy as np
import matplotlib.pyplot as plt
from agent import Tabular_Q_learning, Policy_gradient, Actor_critics
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
	episodes = 1000
	batch_size = 50
	episode = 0
	steps = 0
	losses = []
	reward_sum = []
	test_score = []

	env = gym.make('CartPole-v1')
	agent = Policy_gradient(env)

	states = np.empty(0).reshape(0,agent.dimen)
	actions = np.empty(0).reshape(0,1)
	rewards = np.empty(0).reshape(0,1)
	discounted_rewards = np.empty(0).reshape(0,1)



	while episode < episodes:
		observation = agent.env.reset()
		for step in range(500):
		    state = np.reshape(observation, [1, agent.dimen])
		    
		    predict = agent.model_predict.predict([state])[0]
		    action = np.random.choice(range(agent.num_actions),p=predict)
		    
		    states = np.vstack([states, state])
		    actions = np.vstack([actions, action])
		    
		    observation, reward, done, _ = agent.env.step(action)
		    rewards = np.vstack([rewards, reward])
		    
		    if done:
		        reward_sum.append(step)
		        discounted_rewards_episode = agent.discount_rewards(rewards, agent.gamma)       
		        discounted_rewards = np.vstack([discounted_rewards, discounted_rewards_episode])
		        
		        rewards = np.empty(0).reshape(0,1)

		        if (episode + 1) % batch_size == 0:
		            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / discounted_rewards.std()
		            discounted_rewards = discounted_rewards.squeeze()
		            actions = actions.squeeze().astype(int)
		           
		            actions_train = np.zeros([len(actions), agent.num_actions])
		            actions_train[np.arange(len(actions)), actions] = 1
		            
		            loss = agent.model_train.train_on_batch([states, discounted_rewards], actions_train)
		            losses.append(loss)

		            states = np.empty(0).reshape(0,agent.dimen)
		            actions = np.empty(0).reshape(0,1)
		            discounted_rewards = np.empty(0).reshape(0,1)


		        score = agent.score_model(agent.model_predict,1)
		        test_score.append(score)
		        every = 100
		        if (episode + 1) % every == 0:
		            print("Avg steps for episode {}/{}: {:0.2f} Test Score: {:0.2f} Loss: {:0.6f} ".format(
		                (episode + 1), episodes, sum(reward_sum[-every:])/every, 
		                test_score[-1], np.mean(losses[-every:])))
        
		        episode += 1
		        break
	plt.plot(test_score)
	plt.show()


if __name__ == '__main__':
	# agent_Q()
	agent_pg()
