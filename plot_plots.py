import pickle
import matplotlib.pyplot as plt

def plot1():
	with open('Q_scores.pickle','rb') as f:
	    Q_scores = pickle.load(f)

	# plt.plot(Q_scores)
	# plt.show()

	with open('pg_scores.pickle','rb') as f:
	    pg_scores = pickle.load(f)

	# plt.plot(pg_scores)
	# plt.show()

	with open('ac_scores.pickle','rb') as f:
	    ac_scores = pickle.load(f)


	plt.plot(range(2000), Q_scores, label='Tabular Q learning')
	plt.plot(range(2000), pg_scores, label='Policy gradient')
	plt.plot(range(2000), ac_scores, label='Actor-critic')
	plt.xlabel('Episode')
	plt.ylabel('Score')
	plt.legend()
	plt.show()

def plot2():
	with open('Q_scores.pickle','rb') as f:
	    Q_scores = pickle.load(f)

	with open('Q_scores_1e2.pickle','rb') as f:
	    Q_scores_1e2 = pickle.load(f)
	with open('Q_scores_high.pickle','rb') as f:
	    Q_scores_high = pickle.load(f)
	with open('Q_scores_no_penalty.pickle','rb') as f:
	    Q_scores_no_penalty = pickle.load(f)

	plt.plot(range(2000), Q_scores_high, label='with penalty -1e6')
	plt.plot(range(2000), Q_scores, label='with penalty -1e5')
	plt.plot(range(2000), Q_scores_1e2, label='with penalty -1e2')
	plt.plot(range(2000), Q_scores_no_penalty, label='without penalty')
	plt.xlabel('Episode')
	plt.ylabel('Score')
	plt.title('Tabular Q learning')
	plt.legend()
	plt.show()


def plot3():
	with open('pg_scores.pickle','rb') as f:
	    pg_scores = pickle.load(f)
	with open('pg_scores_no_stdize.pickle','rb') as f:
	    pg_scores_no_stdize = pickle.load(f)

	plt.plot(range(2000), pg_scores, label='standardized')
	plt.plot(range(2000), pg_scores_no_stdize, label='not standardized')
	# plt.plot(range(2000), Q_scores_1e2, label='with penalty -1e2')
	# plt.plot(range(2000), Q_scores_no_penalty, label='without penalty')
	plt.xlabel('Episode')
	plt.ylabel('Score')
	plt.title('Policy gradient')
	plt.legend()
	plt.show()

def plot4():
	with open('pg_scores.pickle','rb') as f:
	    pg_scores = pickle.load(f)
	with open('pg_scores_default_init.pickle','rb') as f:
	    pg_scores_default_init = pickle.load(f)

	plt.plot(range(2000), pg_scores, label='init with ones')
	plt.plot(range(2000), pg_scores_default_init, label='default init')
	# plt.plot(range(2000), Q_scores_1e2, label='with penalty -1e2')
	# plt.plot(range(2000), Q_scores_no_penalty, label='without penalty')
	plt.xlabel('Episode')
	plt.ylabel('Score')
	plt.title('Policy gradient')
	plt.legend()
	plt.show()

def plot5():
	with open('pg_scores.pickle','rb') as f:
	    pg_scores = pickle.load(f)
	with open('pg_scores_no_dropout.pickle','rb') as f:
	    pg_scores_no_dropout = pickle.load(f)

	plt.plot(range(2000), pg_scores, label='dropout 0.6')
	plt.plot(range(2000), pg_scores_no_dropout, label='no dropout')
	# plt.plot(range(2000), Q_scores_1e2, label='with penalty -1e2')
	# plt.plot(range(2000), Q_scores_no_penalty, label='without penalty')
	plt.xlabel('Episode')
	plt.ylabel('Score')
	plt.title('Policy gradient')
	plt.legend()
	plt.show()

def plot6():
	with open('pg_scores.pickle','rb') as f:
	    pg_scores = pickle.load(f)
	with open('pg_scores_batch_1.pickle','rb') as f:
	    pg_scores_batch_1 = pickle.load(f)

	plt.plot(range(2000), pg_scores, label='batchsize=50')
	plt.plot(range(2000), pg_scores_batch_1, label='batchsize=1')
	# plt.plot(range(2000), Q_scores_1e2, label='with penalty -1e2')
	# plt.plot(range(2000), Q_scores_no_penalty, label='without penalty')
	plt.xlabel('Episode')
	plt.ylabel('Score')
	plt.title('Policy gradient')
	plt.legend()
	plt.show()

plot6()