from stable_baselines.common.vec_env import SubprocVecEnv
import pommerman
from pommerman import agents

def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        agent_list = []
        for i in range(4):
            agent_list.append(agents.RandomAgent())

        env = pommerman.make(env_id, agent_list)
        env.seed(seed + rank)
        return env

    return _init

num_cpu = 8

env = SubprocVecEnv([make_env("PommeFFACompetition-v0", i) for i in range(num_cpu)])

obs = env.reset()
for _ in range(1000):
    # action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step()
    env.render()