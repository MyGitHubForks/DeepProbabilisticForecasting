from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import pandas as pd
import pickle


def get_adjacency_matrix(adjacency_df, sensor_ids, normalized_k=0.1):
    """

    :param adjacency_df: data frame with two columns: [joint1, joint2].
    :param sensor_ids: list of sensor ids.
    :return:
    """
    num_sensors = len(sensor_ids)
    adj_mat = np.zeros((num_sensors, num_sensors), dtype=np.float32)
    adj_mat[:] = np.inf
    # Builds sensor id to index map.
    sensor_id_to_ind = {}
    for i, sensor_id in enumerate(sensor_ids):
        sensor_id_to_ind[sensor_id] = i

    # Fills cells in the matrix with distances.
    for row in adjacency_df.values:
        if row[0].replace(" ", "") not in sensor_id_to_ind:
            assert False, "bad joint name in adjacency file, ({})".format(row[0])
        if row[1].replace(" ", "") not in sensor_id_to_ind:
            assert False, "bad joint name in adjacency file, ({})".format(row[1])
        adj_mat[sensor_id_to_ind[row[0].replace(" ", "")], sensor_id_to_ind[row[1].replace(" ", "")]] = 1

    # Calculates the standard deviation as theta.
    # distances = adj_mat[~np.isinf(adj_mat)].flatten()
    # std = distances.std()
    # adj_mx = np.exp(-np.square(adj_mat / std))
    # Make the adjacent matrix symmetric by taking the max.
    # adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])

    # Sets entries that lower than a threshold, i.e., k, to zero for sparsity.
    # adj_mx[adj_mx < normalized_k] = 0
    return sensor_ids, sensor_id_to_ind, adj_mat


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sensor_ids_filename', type=str, default='../data/sensor_graph/skeleton_sensor_ids.txt',
                        help='File containing sensor ids separated by comma.')
    parser.add_argument('--adjacency_filename', type=str, default='../data/sensor_graph/skeleton_joint_pairs.csv',
                        help='CSV file containing joint pairs that are connected: [joint1, joint2].')
    parser.add_argument('--output_pkl_filename', type=str, default='../data/sensor_graph/skeleton_adj_mat.pkl',
                        help='Path of the output file.')
    args = parser.parse_args()

    with open(args.sensor_ids_filename) as f:
        sensor_ids = f.read().replace(" ", "").split(',')
    adjacency_df = pd.read_csv(args.adjacency_filename, dtype={'joint1': 'str', 'joint2': 'str'})
    _, sensor_id_to_ind, adj_mx = get_adjacency_matrix(adjacency_df, sensor_ids)
    # Save to pickle file.
    with open(args.output_pkl_filename, 'wb') as f:
        pickle.dump([sensor_ids, sensor_id_to_ind, adj_mx], f, protocol=2)
