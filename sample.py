import gym
import numpy as np
import matplotlib.pyplot as plt
from agent import Tabular_Q_learning, Policy_gradient, Actor_critic
from copy import deepcopy


def agent_Q():
	episodes = 2000
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
		    	reward_hist[episode] = step+1
		    	env.close()
		    	break

		if episode % 1 == 0:
			# print('Episode: {}/{} | Score: {}'.format(episode, episodes, step+1))
			pass


	# print('avg. {}'.format(np.mean(reward_hist)))
	# print('last 100 {}'.format(np.mean(reward_hist[-100:])))
	# print('peak {}'.format(np.max(reward_hist)))
	fig1 = plt.figure(1)
	plt.clf()
	plt.plot(reward_hist)
	plt.xlabel('Epoch')
	plt.ylabel('Score')
	plt.title('Tabular Q Learning')
	# plt.legend()
	plt.savefig("Q_loss.png")
	# plt.show()

	return reward_hist



def agent_pg():
	episodes = 2000
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
		        reward = -10 # 1600
		    agent.rewards = np.vstack([agent.rewards, reward])
		    
		    if done:
		        reward_hist.append(step)

		        discounted_rewards_episode = agent.discount_rewards(agent.rewards)       
		        agent.discounted_rewards = np.vstack([agent.discounted_rewards, discounted_rewards_episode])
		        agent.rewards = np.empty(0).reshape(0,1)

		        if (episode + 1) % batch_size == 0:
		        	loss = agent.train(agent.states, agent.actions, agent.discounted_rewards)
		        	losses.append(loss)
		        
		        if episode % 1 == 0:
		            score = agent.test(1)
		            test_score.append(score)
		            # print("Episode: {}/{} | Score: {:0.2f}".format(episode, episodes, test_score[-1]))
        
		        episode += 1
		        break

	fig2 = plt.figure(2)
	plt.clf()
	plt.plot(test_score)
	plt.xlabel('Epoch')
	plt.ylabel('Score')
	plt.title('Policy Gradient')
	# plt.legend()
	plt.savefig("pg_loss.png")
	return test_score


def agent_ac():
    episodes = 2000
    env = gym.make('CartPole-v1')
    agent = Actor_critic(env)
    reward_hist = []

    for episode in range(episodes):
        observation = agent.env.reset()

        for step in range(500):
            state = observation.reshape(-1, 4)
            action = agent.act(state)

            observation_next, reward, done, _ = agent.env.step(action)
            state_next = observation_next.reshape(-1, 4)
            if done and step < 499:
                reward = -1e-5
            loss1, loss2 = agent.train(state, action, reward, state_next, done)

            observation = state_next[0]

            if done:
                reward_hist.append(step+1)
                if episode % 50 == 0:
                    print('Episode: {}/{} | Score: {}'.format(episode, episodes, step+1))
                break

    plt.plot(reward_hist)
    fig3 = plt.figure(3)
    plt.clf()
    plt.plot(reward_hist)
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Actor-critic')
    # plt.legend()
    plt.savefig("ac_loss.png")
    return reward_hist

if __name__ == '__main__':
	tests = 10
	Q_scores = []
	pg_scores = []
	ac_score = []

	for test in range(tests)
		print('=========================<{}>==========================='.format(test))
		Q_scores.append(agent_Q())

	f = open('Q_scores.pickle', 'wb')
	pickle.dump(Q_scores, f)
	f.close()

	for test in range(tests)
		print('=========================<{}>==========================='.format(test))
		pg_scores.append(agent_pg())

	f = open('pg_scores.pickle', 'wb')
	pickle.dump(pg_scores, f)
	f.close()

	for test in range(tests)
		print('=========================<{}>==========================='.format(test))
		ac_score.append(agent_ac())

	f = open('ac_scores.pickle', 'wb')
	pickle.dump(ac_scores, f)
	f.close()
