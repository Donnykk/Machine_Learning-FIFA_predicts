import os
import warnings

import matplotlib as mpl
import pandas as pd
from sklearn.preprocessing import scale

mpl.use('TkAgg')

warnings.filterwarnings('ignore')

path = './data/'  # data path
res = []
file_list = []

# overall view of files
for root, dirs, files in os.walk(path):
    files.sort()
    for i, file in enumerate(files):
        if os.path.splitext(file)[1] == '.csv':
            file_list.append(file)
            res.append('raw_data_' + str(i + 1))

time_list = [file_list[i][0:4] for i in range(len(file_list))]

# read in data
for i in range(len(res)):
    res[i] = pd.read_csv(path + file_list[i], encoding='ISO-8859-1', on_bad_lines='skip')

# data process
for i in range(len(res), 0, -1):
    if res[i - 1].shape[0] != 380:
        key = 'res[' + str(i) + ']'
        res.pop(i - 1)
        time_list.pop(i - 1)
        continue
columns_req = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
playing_statistics = []
playing_data = {}
for i in range(len(res)):
    playing_statistics.append('playing_statistics_' + str(i + 1))
    playing_statistics[i] = res[i][columns_req]


# feature constructions
def get_goals_diff(playing_stat):
    teams = {}
    for i in playing_stat.groupby('HomeTeam').mean().T.columns:
        teams[i] = []
    for i in range(len(playing_stat)):
        # home court goals score
        HTGS = playing_stat.iloc[i]['FTHG']
        # away from home court goals score
        ATGS = playing_stat.iloc[i]['FTAG']
        teams[playing_stat.iloc[i].HomeTeam].append(HTGS - ATGS)
        teams[playing_stat.iloc[i].AwayTeam].append(ATGS - HTGS)
    goalsDifference = pd.DataFrame(data=teams, index=[i for i in range(1, 39)]).T
    goalsDifference[0] = 0
    for i in range(2, 39):
        goalsDifference[i] = goalsDifference[i] + goalsDifference[i - 1]
    return goalsDifference


def get_gss(playing_stat):
    goalsDifference = get_goals_diff(playing_stat)
    j = 0
    HTGD = []
    ATGD = []
    for i in range(380):
        ht = playing_stat.iloc[i].HomeTeam
        at = playing_stat.iloc[i].AwayTeam
        HTGD.append(goalsDifference.loc[ht][j])
        ATGD.append(goalsDifference.loc[at][j])
        if ((i + 1) % 10) == 0:
            j = j + 1
    playing_stat.loc[:, 'HTGD'] = HTGD
    playing_stat.loc[:, 'ATGD'] = ATGD
    return playing_stat


for i in range(len(playing_statistics)):
    playing_statistics[i] = get_gss(playing_statistics[i])


def get_points(result):
    if result == 'W':
        return 3
    elif result == 'D':
        return 1
    else:
        return 0


# calculate the total points of the season
def get_total_points(match_res):
    match_points = match_res.applymap(get_points)
    for i in range(2, 39):
        match_points[i] = match_points[i] + match_points[i - 1]
    match_points.insert(column=0, loc=0, value=[0 * i for i in range(20)])
    return match_points


def get_result(playing_stat):
    teams = {}
    for i in playing_stat.groupby('HomeTeam').mean().T.columns:
        teams[i] = []
    for i in range(len(playing_stat)):
        if playing_stat.iloc[i].FTR == 'H':
            teams[playing_stat.iloc[i].HomeTeam].append('W')
            teams[playing_stat.iloc[i].AwayTeam].append('L')
        elif playing_stat.iloc[i].FTR == 'A':
            teams[playing_stat.iloc[i].AwayTeam].append('W')
            teams[playing_stat.iloc[i].HomeTeam].append('L')
        else:
            teams[playing_stat.iloc[i].AwayTeam].append('D')
            teams[playing_stat.iloc[i].HomeTeam].append('D')
    return pd.DataFrame(data=teams, index=[i for i in range(1, 39)]).T


# insert the H and A results into the table
def get_agg_points(playing_stat):
    match_res = get_result(playing_stat)
    cum_pts = get_total_points(match_res)
    HTP = []
    ATP = []
    j = 0
    for i in range(380):
        ht = playing_stat.iloc[i].HomeTeam
        at = playing_stat.iloc[i].AwayTeam
        HTP.append(cum_pts.loc[ht][j])
        ATP.append(cum_pts.loc[at][j])
        if ((i + 1) % 10) == 0:
            j = j + 1
    playing_stat.loc[:, 'HTP'] = HTP
    playing_stat.loc[:, 'ATP'] = ATP
    return playing_stat


for i in range(len(playing_statistics)):
    playing_statistics[i] = get_agg_points(playing_statistics[i])


def get_recent(playing_stat, num):
    recent = get_result(playing_stat)
    recent_final = recent.copy()
    for i in range(num, 39):
        recent_final[i] = ''
        j = 0
        while j < num:
            recent_final[i] += recent[i - j]
            j += 1
    return recent_final


def add_recent(playing_stat, num):
    recent = get_recent(playing_stat, num)
    # M represents unknown
    h = ['M' for _ in range(num * 10)]
    a = ['M' for _ in range(num * 10)]
    j = num
    for i in range((num * 10), 380):
        ht = playing_stat.iloc[i].HomeTeam
        at = playing_stat.iloc[i].AwayTeam
        past = recent.loc[ht][j]
        h.append(past[num - 1])
        past = recent.loc[at][j]
        a.append(past[num - 1])
        if ((i + 1) % 10) == 0:
            j = j + 1
    playing_stat['HM' + str(num)] = h
    playing_stat['AM' + str(num)] = a
    return playing_stat


def add_recent_df(playing_statistics_):
    playing_statistics_ = add_recent(playing_statistics_, 1)
    playing_statistics_ = add_recent(playing_statistics_, 2)
    playing_statistics_ = add_recent(playing_statistics_, 3)
    return playing_statistics_


for i in range(len(playing_statistics)):
    playing_statistics[i] = add_recent_df(playing_statistics[i])


def get_mw(playing_stat):
    j = 1
    MatchWeek = []
    for i in range(380):
        MatchWeek.append(j)
        if ((i + 1) % 10) == 0:
            j = j + 1
    playing_stat['MW'] = MatchWeek
    return playing_stat


for i in range(len(playing_statistics)):
    playing_statistics[i] = get_mw(playing_statistics[i])

# merge data
playing_stat = pd.concat(playing_statistics, ignore_index=True)

# get average HTGD, ATGD ,HTP, ATP
cols = ['HTGD', 'ATGD', 'HTP', 'ATP']
for col in cols:
    playing_stat[col] = playing_stat[col] / playing_stat.MW

# drop some useless data
playing_stat = playing_stat[playing_stat.MW > 3]
playing_stat.drop(['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'MW'], 1, inplace=True)

n_matches = playing_stat.shape[0]
n_home_wins = len(playing_stat[playing_stat.FTR == 'H'])
win_rate = (float(n_home_wins) / n_matches) * 100
# print("home win rate: {:.2f}%".format(win_rate))


# home win or not
def only_hw(string):
    if string == 'H':
        return 'H'
    else:
        return 'NH'


playing_stat['FTR'] = playing_stat.FTR.apply(only_hw)
# Split the features and labels
X_all = playing_stat.drop(['FTR'], 1)
y_all = playing_stat['FTR']


def convert_(data):
    max_ = data.max()
    min_ = data.min()
    return (data - min_) / (max_ - min_)


r_data = convert_(X_all['HTGD'])
cols = [['HTGD', 'ATGD', 'HTP', 'ATP']]
for col in cols:
    X_all[col] = scale(X_all[col])

# feature to string
X_all.HM1 = X_all.HM1.astype('str')
X_all.HM2 = X_all.HM2.astype('str')
X_all.HM3 = X_all.HM3.astype('str')
X_all.AM1 = X_all.AM1.astype('str')
X_all.AM2 = X_all.AM2.astype('str')
X_all.AM3 = X_all.AM3.astype('str')


def preprocess_features(X):
    output = pd.DataFrame(index=X.index)
    for column, col_data in X.iteritems():
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix=column)
        output = output.join(col_data)
    return output


X_all = preprocess_features(X_all)

