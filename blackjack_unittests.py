import numpy as np
import gym
import blackjack
from collections import defaultdict

def _print_success_message():
    print('Tests Passed')

def test_get_policy_for_observation():
    # setup
    env = gym.make('Blackjack-v0')

    agent = blackjack.BlackjackAgent(env.action_space, epsilon_decay=8000)

    Q_s = defaultdict()
    Q_s[(4, 7, False)] = [-0.45, -0.75]

    policy = agent.get_policy_for_observation(Q_s, .1)

    env.close()

    # Check type
    assert isinstance(policy, np.ndarray),\
        'Policy is wrong type. Found {} type.'.format(type(policy))

    # Check value
    np.testing.assert_allclose(policy, [0.95, 0.05], rtol=1e-5, atol=0,\
        err_msg='Policy probability values are wrong')

    _print_success_message()
