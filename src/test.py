import argparse
import numpy as np
from environment import Environment

seed = 100


def parse():
    parser = argparse.ArgumentParser(description="DeepRL Final Project")
    parser.add_argument('--test_dqn', action='store_true', help='Test mode for DQN')
    args = parser.parse_args()
    return args


def test(agent, env, total_episodes=30):
    rewards = []
    env.seed(seed)
    for i in range(total_episodes):
        state = env.reset()
        agent.init_game_setting()
        done = False
        episode_reward = 0.0

        # playing one game
        while not done:
            action = agent.make_action(state, test=True)
            state, reward, done, info = env.step(action)
            episode_reward += reward
        print("rwd", episode_reward)

        rewards.append(episode_reward)
    print('Run %d episodes' % (total_episodes))
    print('Mean:', np.mean(rewards))


def run(args):
    if args.test_dqn:
        env = Environment('BreakoutNoFrameskip-v4', args, test=True)
        from agent_dqn import Agent_DQN
        agent = Agent_DQN(env, args)
        test(agent, env, total_episodes=100)


if __name__ == '__main__':
    args = parse()
    run(args)
