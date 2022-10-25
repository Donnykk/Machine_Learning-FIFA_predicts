import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt

mpl.use('TkAgg')

players_ = pd.read_csv('data/fifa_player_dataset_2019.csv', encoding='ISO-8859-1')

important_columns = ['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physicality', 'rating']

keeper = ['GK']
defend = ['CB', 'LB', 'RB', 'RWB', 'LWB']
midfield = ['CDM', 'CM', 'CAM', 'RM', 'RW', 'LM', 'LW']
attack = ['CF', 'ST', 'RF', 'LF']


def create_positions(data, position_list):
    condition = np.zeros(len(data), dtype=bool)
    for position in position_list:
        condition = condition | (data['position'] == position)
    return data[condition]


goalkeeper_ = create_positions(players_, keeper)
defender_ = create_positions(players_, defend)
midfielder_ = create_positions(players_, midfield)
attacker_ = create_positions(players_, attack)


def graph_fifa_stats(data, COL_NUM=2, ROW_NUM=4):
    fig, axes = plt.subplots(ROW_NUM, COL_NUM, figsize=(12, 12))

    for i, column in enumerate(important_columns):
        ax = axes[int(i / COL_NUM), i % COL_NUM]
        sns.histplot(data[column], ax=ax)
        ax.set_title(column + ' Distribution')
        plt.tight_layout()
    plt.show()


def graph_fifa_stats_overlap(data, data_, COL_NUM=2, ROW_NUM=4):
    fig, axes = plt.subplots(ROW_NUM, COL_NUM, figsize=(12, 12))

    for i, column in enumerate(important_columns):
        ax = axes[int(i / COL_NUM), i % COL_NUM]
        sns.histplot(data[column], ax=ax)
        sns.histplot(data_[column], ax=ax)
        ax.set_title(column + ' Distribution')
        plt.tight_layout()
    plt.show()


# graph_fifa_stats(players_)
# graph_fifa_stats_overlap(players_, data_=goalkeeper_)
# graph_fifa_stats_overlap(players_, data_=defender_)
# graph_fifa_stats_overlap(players_, data_=midfielder_)
# graph_fifa_stats_overlap(players_, data_=attacker_)

def team_averages(fifa_stat):
    fifa_stat['team'] = fifa_stat['team'].apply(lambda x: x.strip())
    team_array = fifa_stat['team'].unique()

    team_list = list()
    for team in team_array:
        team_list.append((team, fifa_stat[fifa_stat['team'] == team]))

    averages_by_team = list()
    for (team, stats) in team_list:
        averages_by_team.append((team, stats['rating'].mean(), stats['pace'].mean(), stats['shooting'].mean(),
                                 stats['passing'].mean(), stats['dribbling'].mean(), stats['defending'].mean(),
                                 stats['physicality'].mean(),))
    df = pd.DataFrame(averages_by_team,
                      columns=['team', 'rating', 'pace', 'shooting', 'passing', 'dribbling', 'defending',
                               'physicality'])
    df = df.set_index('team')
    return df


def team_graphs(team_data, COL_NUM=2, ROW_NUM=4):
    fig, axes = plt.subplots(ROW_NUM, COL_NUM, figsize=(12, 12))
    columns = ['rating', 'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physicality']

    for i, column_name in enumerate(columns):
        ax = axes[int(i / COL_NUM), i % COL_NUM]
        data = team_data[column_name]
        data = data.sort_values(ascending=False)[:5]
        data.plot(kind='barh', ax=ax)
        ax.set_title(column_name)
        ax.set_xlim(50, 80)

    plt.tight_layout()
    plt.show()


team_ = team_averages(players_)
team_graphs(team_)

forward = pd.read_csv('forward.csv', encoding='ISO-8859-1')
midfielders = pd.read_csv('midfielder.csv', encoding='ISO-8859-1')
defender = pd.read_csv('defender.csv', encoding='ISO-8859-1')
goalkeeper = pd.read_csv('goalkeeper.csv', encoding='ISO-8859-1')

print("Top goalscorers for the 2019 season")
print(forward[['name', 'team', 'rating', 'goals']].nlargest(10, 'goals'))
print('\n\n')

print("Top assists for the 2019 season")
print(midfielders[['name', 'team', 'rating', 'assists']].nlargest(10, 'assists'))
print('\n\n')

print("Top tackles for the 2019 season")
print(defender[['name', 'team', 'rating', 'tackles']].nlargest(10, 'tackles'))
print('\n\n')

print("Top goalkeepers for the 2019 season")
print(goalkeeper[['name', 'team', 'rating', 'saves']].nlargest(10, 'saves'))
