import pickle
import numpy as np
from matplotlib import pyplot as plot

# ------ Hyper Parameters -----
sd_range_side = 10
# -----------------------------

# We create three reward lists
reward_list_5 = 0
reward_list_10 = 0
reward_list_40 = 0

# Reading in the pickle files to load for with ae
with open("with_ae_4_rovers_10_POI_5_latent_rewards_p3.pkl", "rb") as file_1:
    # Loads the with ae rewards list
    reward_list_5 = pickle.load(file_1)
# Reading in the pickle files to load for with ae
with open("with_ae_4_rovers_10_POI_5_latent_rewards_p2.pkl", "rb") as file_3:
    # Loads the with ae rewards list
    reward_list_5 += pickle.load(file_3)
# We get the number of rewards
reward_count_5 = len(reward_list_5)

# Reading in the pickle files to load for with ae
with open("with_ae_4_rovers_10_POI_rewards_1.pkl", "rb") as file_1:
    # Loads the with ae rewards list
    reward_list_10 = pickle.load(file_1)
# Reading in the pickle files to load for with ae
with open("with_ae_4_rovers_10_POI_rewards_2.pkl", "rb") as file_2:
    # Loads the with ae rewards list
    reward_list_10 += pickle.load(file_2)
# We get the number of rewards
reward_count_10 = len(reward_list_10)

# Reading in the pickle files to load for without ae
with open("without_ae_4_rovers_10_POI_rewards.pkl", "rb") as file_1:
    # Loads the with ae rewards list
    reward_list_40 = pickle.load(file_1)
# We get the number of rewards
reward_count_40 = len(reward_list_40)

# We limit the length to minimum of three
reward_count = min([reward_count_5, reward_count_10, reward_count_40])

# We restrict the number of episodes to minimum
reward_list_5 = reward_list_5[:reward_count]
reward_list_10 = reward_list_10[:reward_count]
reward_list_40 = reward_list_40[:reward_count]

# We create lists to store the standard deviation
std_list_5 = []
std_list_10 = []
std_list_40 = []

# We iterate through the reward list
for i in range(reward_count):
    local_values_5 = []
    local_values_10 = []
    local_values_40 = []
    for j in range(max(i-sd_range_side, 0), min(i+sd_range_side, reward_count)):
        # Local values are updated
        local_values_5.append(reward_list_5[j])
        local_values_10.append(reward_list_10[j])
        local_values_40.append(reward_list_40[j])
    # Get the standard deviation list updated
    std_list_5.append(np.std(np.array(local_values_5)))
    std_list_10.append(np.std(np.array(local_values_10)))
    std_list_40.append(np.std(np.array(local_values_40)))

# We convert all std lists to numpy
std_list_5 = np.array(std_list_5)
std_list_10 = np.array(std_list_10)
std_list_40 = np.array(std_list_40)

# The episode list
episode_list = [_ for _ in range(reward_count)]

# We obtain the trend line
trend_line_5 = np.poly1d(np.polyfit(episode_list, reward_list_5, 9))
trend_line_10 = np.poly1d(np.polyfit(episode_list, reward_list_10, 9))
trend_line_40 = np.poly1d(np.polyfit(episode_list, reward_list_40, 9))

# plots the two rewards over episodes
plot.plot(episode_list, trend_line_5(episode_list), color=[0.5, 0, 0], linewidth=3)

plot.fill_between(episode_list, np.subtract(trend_line_5(episode_list), std_list_5),
                  np.add(trend_line_5(episode_list), std_list_5), facecolor=[1, 0, 0], alpha=0.13)

plot.plot(episode_list, trend_line_10(episode_list), color=[0, 0.5, 0], linewidth=3)

plot.fill_between(episode_list, np.subtract(trend_line_10(episode_list), std_list_10),
                  np.add(trend_line_10(episode_list), std_list_10), facecolor=[0, 1, 0], alpha=0.13)

plot.plot(episode_list, trend_line_40(episode_list), color=[0, 0, 0.5], linewidth=3)

plot.fill_between(episode_list, np.subtract(trend_line_40(episode_list), std_list_40),
                  np.add(trend_line_40(episode_list), std_list_40), facecolor=[0, 0, 1], alpha=0.13)

plot.plot(episode_list, [10 for _ in episode_list], color=[0, 0, 0], linewidth=1.5, linestyle='--')

# We mark the label on the x-axis
plot.xlabel("Training Episodes")
# We mark the label on the y-axis
plot.ylabel('Global Reward')
# The plot grid is highlighted
plot.grid()
# The plot title is displayed
plot.title("Global Rewards Across Episodes: DDPG Trained on 4 Rovers and 10 POI")
# Marking the plot legends
plot.gca().legend(('Compressed State Space of 5', 'Compressed State Space of 10',
                   'Original State Space of 40', 'Maximum Attainable Reward'))
# Setting the plot limits
plot.xlim((0, reward_count-1))
# We plot the image
plot.show()
