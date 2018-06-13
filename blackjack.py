import argparse
import sys
import logging
import numpy as np
from collections import defaultdict

import gym
from gym import wrappers, logger

GAMMA = 1.0
NUM_EPISODES = 500000

logger = logging.getLogger()


class BlackjackAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space
        self.nA = action_space.n
        # initialize Q value and N count dictionaries
        self.Q = defaultdict(lambda: np.zeros(action_space.n))
        self.N = defaultdict(lambda: np.zeros(action_space.n))
        # track episode num for epsilon
        self.i_episode = 0

    def log_observation(self, observation):
        player_hand_sum, dealer_showing_card, usable_ace = zip(observation)
        logger.debug('players current sum:{}, dealer showing card:{}, usable ace:{}'.format(player_hand_sum[0], dealer_showing_card[0], usable_ace[0]))

    def log_done(self, observation, reward):
        self.log_observation(observation)
        logger.debug('final reward:{}\n'.format(reward))

    def get_policy_for_observation(self, Q_s, epsilon):
        """ calculates the action probabilities corresponding to epsilon-greedy policy """
        policy_s = np.ones(self.nA) * epsilon / self.nA
        best_a = np.argmax(Q_s)
        policy_s[best_a] = 1 - epsilon + (epsilon / self.nA)
        return policy_s

    def choose_action(self, observation, epsilon):
        """ if observation in Q dict, choose random action with probability of eps otherwise sample from action space """
        if observation in self.Q:
            action = np.random.choice(np.arange(self.nA), p=self.get_policy_for_observation(self.Q[observation], epsilon))
        else:
            action = self.action_space.sample()
        return action

    def update_action_val_function(self, episode):
        """ updates the action-value function Q and N count dictionaries for one episode """
        observations, actions, rewards = zip(*episode)
        discounts = np.array([GAMMA**i for i in range(len(rewards)+1)])
        for i, observation in enumerate(observations):
            old_Q = self.Q[observation][actions[i]] 
            old_N = self.N[observation][actions[i]]
            self.Q[observation][actions[i]] = old_Q + (sum(rewards[i:]*discounts[:-(1+i)]) - old_Q)/(old_N+1)
            self.N[observation][actions[i]] += 1

    def generate_episode(self, env):
        """ start new episode and update action-value function until episode terminates """
        self.i_episode += 1
        # decay epsilon
        epsilon = 1.0/((self.i_episode/8000) + 1)
        episode = []
        observation = env.reset()
        self.log_observation(observation)
        while True:
            action = self.choose_action(observation, epsilon)
            logger.debug('HIT' if action else 'STICK')
            next_observation, reward, done, _ = env.step(action)
            episode.append((observation, action, reward))
            if done:
                self.log_done(next_observation, reward)
                break
            else:
                self.log_observation(next_observation)
            observation = next_observation
        return episode

def learn(base_dir='blackjack-1'):
    env = gym.make('Blackjack-v0')
    env = wrappers.Monitor(env, directory=base_dir, force=True, video_callable=False)

    agent = BlackjackAgent(env.action_space)

    for i in range(NUM_EPISODES):
        if i % 1000 == 0:
            logger.debug('\rEpisode {}/{}.'.format(i, NUM_EPISODES))
        episode = agent.generate_episode(env)
        agent.update_action_val_function(episode)

    # obtain the policy from the action-value function
    policy = dict((k,np.argmax(v)) for k, v in agent.Q.items())

    env.close()

    return policy

def choose_action_by_policy(action_space, policy, observation):
    """ selects action based on trained policy """
    if observation in policy:
        action = policy[observation]
    else:
        # observation not found in policy, usually if there hasn't been enough training episodes
        action = action_space.sample()
    return action

def score(policy, num_episodes=1000):
    """ average score using policy after num_episodes """
    env = gym.make('Blackjack-v0')
    rewards = []
    for _ in range(num_episodes):
        observation = env.reset()
        rewards_sum = 0
        while True:
            action = choose_action_by_policy(env.action_space, policy, observation)
            next_observation, reward, done, _ = env.step(action)
            rewards_sum += reward
            if done:
                rewards.append(rewards_sum)
                break
            observation = next_observation
    env.close()

    return np.mean(rewards)

def main():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-b', '--base-dir', default='blackjack-1', help='Set base dir.')
    parser.add_argument('-v', '--verbose', action='count', dest='verbosity', default=0, help='Set verbosity.')    
    args = parser.parse_args()

    if args.verbosity == 0:
        logger.setLevel(logging.INFO)
    elif args.verbosity >= 1:
        logger.setLevel(logging.DEBUG)
    
    policy = learn(base_dir=args.base_dir)

    final_average_return = score(policy)
    logger.info("final average returns: {}".format(final_average_return))

    # TODO: plot the policy
    # plot_blackjack_values(V)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

