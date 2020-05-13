import pommerman
from pommerman import agents
import os


def main():
    print(pommerman.REGISTRY)

    candidates = [
        # 'simple',
        'eisenach',
        # 'navocado',
        'hakozakijunctions',
        # 'skynet955'
    ]

    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            if candidates[i] == 'simple':
                agent_list = [
                    agents.SimpleAgent(),
                    agents.DockerAgent('multiagentlearning/' + candidates[j], port=11111),
                    agents.SimpleAgent(),
                    agents.DockerAgent('multiagentlearning/' + candidates[j], port=11113),
                ]
            else:
                agent_list = [
                    agents.DockerAgent('multiagentlearning/' + candidates[i], port=22220),
                    agents.DockerAgent('multiagentlearning/' + candidates[j], port=22221),
                    agents.DockerAgent('multiagentlearning/' + candidates[i], port=22222),
                    agents.DockerAgent('multiagentlearning/' + candidates[j], port=22223),
                ]
            # Make the "Free-For-All" environment using the agent list
            env = pommerman.make('PommeTeamCompetitionFast-v0', agent_list)

            wins = 0
            ties = 0
            survived_agents = []
            nof_plays = 100
            # Run the episodes just like OpenAI Gym
            for i_episode in range(nof_plays):
                print("Game " + str(i_episode))
                state = env.reset()
                # if i_episode != 20:
                #    continue
                done = False
                while not done:
                    # env.render()
                    actions = env.act(state)
                    state, reward, done, info = env.step(actions)

                print(info)
                if info['result'] == pommerman.constants.Result.Tie:
                    ties += 1
                elif info['winners'] == [1, 3]:
                    wins += 1
                else:
                    print(info['result'])
                    print(info['winners'])
                    print("Lost with seed: " + str(i_episode))
                print('Episode {} finished'.format(i_episode))
                survived_agents.extend(state[0]['alive'])
            env.close()

            survived_team_0 = survived_agents.count(10) + survived_agents.count(12)
            survived_team_1 = survived_agents.count(11) + survived_agents.count(13)
            kills = nof_plays * 2 - survived_team_0
            death = nof_plays * 2 - survived_team_1
            print("kills / death / ratio: ", kills, " / ", death, " / ", kills / max(0.1, death))
            winRatio = str(wins / max(1, (nof_plays - ties)))
            print("wins: " + str(wins) + "/" + str(nof_plays - ties) + " = " + winRatio)

            file = open("/home/lucius/pommerman_results/winrate_{}vs{}.txt".format(candidates[i], candidates[j]), "w")
            file.write('{}\t{}\t{}\t{}'.format(wins, ties, nof_plays - ties - wins, winRatio))
            file.close()
            file = open("/home/lucius/pommerman_results/kd_{}vs{}.txt".format(candidates[i], candidates[j]), "w")
            file.write("{}\t{}".format(kills, death))
            file.close()

            for agent in agent_list:
                if isinstance(agent, agents.DockerAgent):
                    agent.shutdown()


if __name__ == '__main__':
    print(os.getpid())
    import time

    # time.sleep(10)
    main()
