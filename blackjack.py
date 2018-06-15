import argparse
import sys
import logging
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable

import gym
from gym import wrappers, logger

# Rewards in Blackjack happen at the end of the game, no discounting
GAMMA = 1.0

logger = logging.getLogger()


class BlackjackAgent(object):
    def __init__(self, action_space, epsilon_decay):
        self.action_space = action_space
        self.nA = action_space.n
        self.epsilon_decay = epsilon_decay

        # initialize Q value and N count dictionaries
        self.Q = defaultdict(lambda: np.zeros(action_space.n))
        self.N = defaultdict(lambda: np.zeros(action_space.n))
        # track episode num for epsilon
        self.i_episode = 0

    def log_observation(self, observation):
        player_hand, dealer_showing, usable_ace = zip(observation)
        logger.debug('player hand:{}, dealer showing:{}, usable ace:{}'.format(player_hand[0], dealer_showing[0], usable_ace[0]))

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
        """ updates the action-value function Q and N count dictionaries for every observation in one episode """
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
        epsilon = 1.0/((self.i_episode/self.epsilon_decay) + 1)
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

def learn(base_dir='blackjack-1', num_episodes=100000, epsilon_decay=8000):
    env = gym.make('Blackjack-v0')
    env = wrappers.Monitor(env, directory=base_dir, force=True, video_callable=False)

    agent = BlackjackAgent(env.action_space, epsilon_decay)

    for i in range(num_episodes):
        if i % 1000 == 0:
            logger.debug('\rEpisode {}/{}.'.format(i, num_episodes))
        episode = agent.generate_episode(env)
        agent.update_action_val_function(episode)

    # obtain the policy from the action-value function
    # e.g. generate  ((4, 7, False), 1)   HIT      ((18, 6, False), 0)  STICK
    policy = dict((k, np.argmax(v)) for k, v in agent.Q.items())

    env.close()

    return policy, agent.Q

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

def plot_policy(policy, plot_filename="plot.png"):

    def get_Z(player_hand, dealer_showing, usable_ace):
        if (player_hand, dealer_showing, usable_ace) in policy:
            return policy[player_hand, dealer_showing, usable_ace]
        else:
            return 1

    def get_figure(usable_ace, ax):
        x_range = np.arange(1, 11)
        y_range = np.arange(11, 22)
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.array([[get_Z(player_hand, dealer_showing, usable_ace) for dealer_showing in x_range] for player_hand in range(21, 10, -1)])
        surf = ax.imshow(Z, cmap=plt.get_cmap('Accent', 2), vmin=0, vmax=1, extent=[0.5, 10.5, 10.5, 21.5])
        plt.xticks(x_range, ('A', '2', '3', '4', '5', '6', '7', '8', '9', '10'))
        plt.yticks(y_range)
        ax.set_xlabel('Dealer Showing')
        ax.set_ylabel('Player Hand')
        ax.grid(color='black', linestyle='-', linewidth=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(surf, ticks=[0, 1], cax=cax)
        cbar.ax.set_yticklabels(['0 (STICK)','1 (HIT)'])
        cbar.ax.invert_yaxis() 
            
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(121)
    ax.set_title('Usable Ace', fontsize=16)
    get_figure(True, ax)
    ax = fig.add_subplot(122)
    ax.set_title('No Usable Ace', fontsize=16)
    get_figure(False, ax)
    plt.savefig(plot_filename)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-b', '--base-dir', default='blackjack-1', help='Set base dir.')
    parser.add_argument('-v', '--verbose', action='count', dest='verbosity', default=0, help='Set verbosity.')    
    args = parser.parse_args()

    if args.verbosity == 0:
        logger.setLevel(logging.INFO)
    elif args.verbosity >= 1:
        logger.setLevel(logging.DEBUG)
    
    num_episodes = 100000
    epsilon_decay = 8000

    policy, Q = learn(args.base_dir, num_episodes, epsilon_decay)

    final_average_return = score(policy)
    logger.info("final average returns: {}".format(final_average_return))

    plot_policy(policy, "diag_{}_{}_{}.png".format(num_episodes, epsilon_decay, final_average_return))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

