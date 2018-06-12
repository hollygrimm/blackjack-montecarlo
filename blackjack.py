import sys
import gym
from gym import wrappers, logger
import numpy as np
from collections import defaultdict

class BlackjackAgent(object):
    def __init__(self, action_space, gamma):
        self.action_space = action_space
        self.nA = action_space.n
        self.GAMMA = gamma
        # initialize Q value and N count dictionaries
        self.Q = defaultdict(lambda: np.zeros(action_space.n))
        self.N = defaultdict(lambda: np.zeros(action_space.n))
        # track episode num for episilon
        self.i_episode = 0

    def log_state(self, state):
        player_hand_sum, dealer_showing_card, usable_ace = zip(state)
        logger.info('players current sum:{}, dealer showing card:{}, usable ace:{}'.format(player_hand_sum[0], dealer_showing_card[0], usable_ace[0]))

    def log_done(self, state, reward):
        self.log_state(state)
        logger.info('final reward:{}'.format(reward))

    def get_policy_for_state(self, Q_s, epsilon):
        """ calculates the action probabilities corresponding to epsilon-greedy policy """
        policy_s = np.ones(self.nA) * epsilon / self.nA
        best_a = np.argmax(Q_s)
        policy_s[best_a] = 1 - epsilon + (epsilon / self.nA)
        return policy_s

    def choose_action_by_policy(self, policy, state):
        """ selects action based on trained policy """
        if state in policy:
            action = policy[state]
        else:
            # state not found in policy, usually if there hasn't been enough training episodes
            action = self.action_space.sample()
        return action

    def choose_action(self, state, epsilon):
        """ if state in Q dict, choose random action with probability of eps otherwise sample from action space """
        if state in self.Q:
            action = np.random.choice(np.arange(self.nA), p=self.get_policy_for_state(self.Q[state], epsilon))
        else:
            action = self.action_space.sample()
        return action

    def update_action_val_function(self, episode):
        """ updates the action-value function Q and N count dictionaries for one episode """
        states, actions, rewards = zip(*episode)
        discounts = np.array([self.GAMMA**i for i in range(len(rewards)+1)])
        for i, state in enumerate(states):
            old_Q = self.Q[state][actions[i]] 
            old_N = self.N[state][actions[i]]
            self.Q[state][actions[i]] = old_Q + (sum(rewards[i:]*discounts[:-(1+i)]) - old_Q)/(old_N+1)
            self.N[state][actions[i]] += 1

    def generate_episode(self, env):
        """ start new episode and update action-value function until episode terminates """
        self.i_episode += 1
        # decay epsilon
        epsilon = 1.0/((self.i_episode/8000) + 1)
        episode = []
        state = env.reset()
        self.log_state(state)
        while True:
            action = self.choose_action(state, epsilon)
            logger.info('HIT' if action else 'STICK')
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                self.log_done(next_state, reward)
                break
            else:
                self.log_state(next_state)
            state = next_state
        return episode

    def score(self, env, policy, num_episodes=1000):
        """ average score using policy after num_episodes """
        rewards = []
        for _ in range(num_episodes):
            state = env.reset()
            rewards_sum = 0
            while True:
                action = self.choose_action_by_policy(policy, state)
                next_state, reward, done, _ = env.step(action)
                rewards_sum += reward
                if done:
                    rewards.append(rewards_sum)
                    break
                state = next_state
        return np.mean(rewards)


if __name__ == "__main__":
    outdir = 'blackjack-1'
    gamma = 1.0
    num_episodes = 1000 # 500000

    # requires version 0.9.7 of gym
    # FIXME logging
    logger.set_level(logger.WARN)

    env = gym.make('Blackjack-v0')
    env = wrappers.Monitor(env, directory=outdir, force=True)
    agent = BlackjackAgent(env.action_space, gamma)

    for i in range(num_episodes):
        if i % 1000 == 0:
            logger.info('\rEpisode {}/{}.'.format(i, num_episodes))
        episode = agent.generate_episode(env)
        agent.update_action_val_function(episode)

    # obtain the policy from the action-value function
    policy = dict((k,np.argmax(v)) for k, v in agent.Q.items())

    final_average_return = agent.score(env, policy)
    print("final average returns: {}".format(final_average_return))

    logger.info(policy[(10, 3, False)])
    logger.info(agent.Q[(10, 3, False)])

    # TODO: plot the policy
    # plot_blackjack_values(V)

    env.close()




