from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import unicodedata


def strip_accents(text):
    return ''.join(char for char in
                   unicodedata.normalize('NFKD', text)
                   if unicodedata.category(char) != 'Mn')


def fix_data(data):
    data['name'] = data['name'].apply(lambda x: x.strip())
    data['name'] = data['name'].apply(lambda x: strip_accents(x))
    data = data.drop_duplicates(subset='name')
    return data


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


# read in and clean data
df = pd.read_csv('data/fifa_player_dataset_2019.csv', encoding='ISO-8859-1')
df = fix_data(df)

gk = pd.read_csv('data/2018_19_GOALKEEPER_stats.csv', encoding='ISO-8859-1', index_col=False)
gk = fix_data(gk)

dfd = pd.read_csv('data/2018_19_DEFENDER_stats.csv', encoding='ISO-8859-1', index_col=False)
dfd = fix_data(dfd)

mid = pd.read_csv('data/2018_19_MIDFIELDER_stats.csv', encoding='ISO-8859-1', index_col=False)
mid = fix_data(mid)

fw = pd.read_csv('data/2018_19_FORWARD_stats.csv', encoding='ISO-8859-1', index_col=False)
fw = fix_data(fw)

keeper = ['GK']
defend = ['CB', 'LB', 'RB', 'RWB', 'LWB']
midfield = ['CDM', 'CM', 'CAM', 'RM', 'RW', 'LM', 'LW']
attack = ['CF', 'ST', 'RF', 'LF']


def create_positions(data, position_list):
    condition = np.zeros(len(data), dtype=bool)
    for position in position_list:
        condition = condition | (data['position'] == position)
    return data[condition]


goalkeeper = create_positions(df, keeper)
defender = create_positions(df, defend)
midfielder = create_positions(df, midfield)
forward = create_positions(df, attack)


def similar_names(data, merged):
    data_names = data['name']
    merged_names = merged['name']
    for i, name in merged_names.items():
        for second_name in data_names:
            if similar(name, second_name) > 0.5:
                if name.split(' ')[-1] == second_name.split(' ')[-1]:
                    merged_names[i] = second_name
                break
    return merged


gk = similar_names(goalkeeper, gk)
goalkeeper_ = pd.merge(goalkeeper, gk, on='name')
dfd = similar_names(defender, dfd)
defender_ = pd.merge(defender, dfd, on='name')
mid = similar_names(midfielder, mid)
midfielder_ = pd.merge(midfielder, mid, on='name')
fw = similar_names(forward, fw)
forward_ = pd.merge(forward, fw, on='name')

forward_.to_csv('forward.csv')
midfielder_.to_csv('midfielder.csv')
defender_.to_csv('defender.csv')
goalkeeper_.to_csv('goalkeeper.csv')
