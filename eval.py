from agents import *
from envs import *
from utils import *
from gym.wrappers import Monitor
from pommerman import agents

N_CHANNELS = 16


def main():
    n_episodes = 100
    env_id = default_config['EnvID']

    agent_list = [
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent(),
    ]

    env = pommerman.make(env_id, agent_list)
    env.reset()
    output_size = env.action_space.n  # 2

    use_cuda = True
    is_render = True
    model_path = 'models/{}.model'.format(env_id)
    predictor_path = 'models/{}.pred'.format(env_id)
    target_path = 'models/{}.target'.format(env_id)

    gamma = float(default_config['Gamma'])

    agent = RNDAgent(
        N_CHANNELS,
        output_size,
        gamma
    )

    print('Loading Pre-trained model....')
    if use_cuda:
        agent.model.load_state_dict(torch.load(model_path))
        agent.rnd.predictor.load_state_dict(torch.load(predictor_path))
        agent.rnd.target.load_state_dict(torch.load(target_path))
    else:
        agent.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        agent.rnd.predictor.load_state_dict(torch.load(predictor_path, map_location='cpu'))
        agent.rnd.target.load_state_dict(torch.load(target_path, map_location='cpu'))
    print('End load...')

    wins = 0
    losses = 0
    ties = 0

    env = PommeWrapper(env, training_agents=0)

    for i in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            if is_render:
                env.render()

            action = agent.act(torch.from_numpy(obs).unsqueeze(0).float().numpy())
            raw_obs, obs, reward, done, info = env.step(action)

            print('episode_reward', info['episode_reward'])

            if done:
                if info['episode_result'] == constants.Result.Win:
                    wins += 1
                elif info['episode_result'] == constants.Result.Loss:
                    losses += 1
                elif info['episode_result'] == constants.Result.Tie:
                    ties += 1

                print('Result: {} Reward: {}'.format(info['episode_result'], info['episode_reward']))

    print('winrate: {}'.format(wins / n_episodes))
    print('losses: {}'.format(losses / n_episodes))
    print('tie: {}'.format(ties / n_episodes))


if __name__ == '__main__':
    main()
