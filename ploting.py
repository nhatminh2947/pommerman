import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

winrate = pd.read_csv('./data/run-FFA_SimpleAgent-tag-reward_mean_extrinsic_reward.csv', delimiter=',')

print(winrate)

fig, ax = plt.subplots()
ax.plot(winrate['Step'], winrate['Value'].to_numpy(), label='Reward')
ax.set_xlabel('Updates')
ax.set_ylabel('Reward')
ax.legend()
plt.show()
