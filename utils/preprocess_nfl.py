# Copyright 2024-present, Riccardo Benaglia, Angelo Porrello, Pietro Buzzega, Simone Calderara, Rita Cucchiara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json
import os
import sys

conf_path = os.getcwd()
sys.path.append(conf_path)
sys.path.append(os.path.join(conf_path, 'utils'))
import numpy as np
import pandas as pd
from conf import base_path_dataset
from tqdm import tqdm

params = dict(
    n_frames=24,
    n_actors=23,
    seed=42,
    stride=2,
    train_percentage=0.8,
)


def select_n_consecutive_sample(data, n, seed, n_actors, stride=1):
    """
    Select n consecutive samples from the data
    """
    data = data.dropna()
    frames = data['frame.id'].value_counts()[data['frame.id'].value_counts() == n_actors].index
    if len(frames.unique()) < n * stride:
        return None
    if frames.nunique() == n * stride:
        return frames.min(), frames.max()
    np.random.seed(seed)
    sorted_uni_data = frames.values
    sorted_uni_data.sort()
    possible_random_start = sorted_uni_data[:-n * stride+1].copy()
    # remove from the possible random start the ones that are not consecutive
    for i in range(len(possible_random_start)):
        if possible_random_start[i] + n * stride - 1 != sorted_uni_data[i + n * stride - 1]:
            possible_random_start[i] = -1
    possible_random_start = [x for x in possible_random_start if x != -1]
    if len(possible_random_start) == 0:
        return None
    random_start = np.random.choice([x for x in possible_random_start if x != -1])
    return random_start, random_start + n * stride - 1


def main(path, n_frames=24, n_actors=23, seed=42, start_games_idx=0, reduces_games_size=-1, stride=2,
         train_percentage=0.6, save=True, ):
    raw_data_path = os.path.join(path, 'raw_data')
    games = pd.read_csv(os.path.join(raw_data_path, 'games.csv'))
    un_games = games['gameId'].unique()
    un_games.sort()
    traj_total_games = []
    for game in tqdm(un_games[start_games_idx:start_games_idx + reduces_games_size]):
        track = pd.read_csv(os.path.join(raw_data_path, 'tracking_gameId_{}.csv'.format(game)))
        assert 0 not in track['nflId'].unique(), '0 in nflId'
        # give to ball nflId = 0
        track['nflId'] = track['nflId'].fillna(0)
        idxes = track.groupby(['playId'])[['frame.id', 'x', 'y']].apply(lambda x: select_n_consecutive_sample(
            x, n_frames, seed, n_actors, stride=stride)).dropna()
        traj_tot = []
        for play in idxes.index:
            start, end = idxes[play]
            traj = track[(track['playId'] == play) & (track['frame.id'] >= start)
                         & (track['frame.id'] <= end)][['nflId', 'x', 'y']]
            sing_traj = []
            if traj['nflId'].nunique() != n_actors:
                continue
            for nflId in traj['nflId'].unique():
                s = traj[traj['nflId'] == nflId][['x', 'y']].to_numpy()
                assert np.isnan(s).sum() == 0, 'nan in s'
                s = s[::stride]
                sing_traj.append(s)
            # on the last dimension we have the players. the single element of the list has dim (n_frames, 2, n_players)
            stack_traj = np.stack(sing_traj, axis=-1)
            assert np.isnan(stack_traj).sum() == 0, 'nan in stack_traj'
            traj_tot.append(stack_traj)

        traj_tot = np.stack(traj_tot, axis=0)
        traj_total_games.append(traj_tot)
    traj_total_games = np.concatenate(traj_total_games, axis=0)
    now = pd.Timestamp.now().strftime('%Y%m%d%H%M%S')
    parameters = {'n_frames': n_frames, 'n_actors': n_actors, 'seed': seed, 'start_games_idx': start_games_idx,
                  'reduces_games_size': reduces_games_size, 'stride': stride, 'train_percentage': train_percentage}
    if save:
        # train test split randomly
        np.random.seed(seed)
        idxes = np.random.permutation(traj_total_games.shape[0])
        train_idxes = idxes[:int(traj_total_games.shape[0] * train_percentage)]
        test_idxes = idxes[int(traj_total_games.shape[0] * train_percentage):]
        print(f'train: {len(train_idxes)}, test: {len(test_idxes)}')
        np.save(os.path.join(path, f'{now}_nfl_train.npy'), traj_total_games[train_idxes])
        np.save(os.path.join(path, f'{now}_nfl_test.npy'), traj_total_games[test_idxes])
        with open(os.path.join(path, f'{now}_params.json'), 'w') as f:
            json.dump(parameters, f)
    return traj_total_games


def test_replicability(n_times=2):
    path = os.path.join(base_path_dataset(), 'nfl')
    tt = []
    for _ in range(n_times):
        tt.append(main(path, start_games_idx=31, reduces_games_size=3,
                       save=False, **params))
    tt = np.stack(tt, axis=0)
    print(np.all(tt[0] == tt[1]))


if __name__ == '__main__':
    # test_replicability(2)
    path = os.path.join(base_path_dataset(), 'nfl')
    tt = main(path, **params)
