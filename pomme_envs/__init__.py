from pommerman import envs, constants, characters
import gym


def blank_env():
    env = envs.v0.Pomme
    game_type = constants.GameType.Team
    env_entry_point = 'pommerman.envs.blank_env:Pomme'
    env_id = 'Blank-PommeTeam-v0'
    env_kwargs = {
        'game_type': game_type,
        'board_size': constants.BOARD_SIZE,
        'num_rigid': constants.NUM_RIGID,
        'num_wood': 0,
        'num_items': 0,
        'max_steps': constants.MAX_STEPS,
        'render_fps': constants.RENDER_FPS,
        'env': env_entry_point,
    }
    agent = characters.Bomber
    return locals()


config = blank_env()
gym.envs.registration.register(id=config['env_id'],
                               entry_point=config['env_entry_point'],
                               kwargs=config['env_kwargs'])
