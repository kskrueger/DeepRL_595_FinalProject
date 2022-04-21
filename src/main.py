import argparse
from agent_dqn import Agent_DQN
from panda_env import PandaEnv
from test import test


def parse():
    parser = argparse.ArgumentParser(description="DS595/CS525 RL Project 4")
    parser.add_argument('--train_dqn', action='store_true', help='whether train DQN')
    parser.add_argument('--test_dqn', action='store_true', help='whether test DQN')

    args = parser.parse_args()
    return args


def run(args):
    env = PandaEnv()
    if args.train_dqn:
        agent = Agent_DQN(env, args)
        agent.train()

    if args.test_dqn:
        agent = Agent_DQN(env, args)
        test(agent, env, total_episodes=100)


if __name__ == '__main__':
    args = parse()
    run(args)
