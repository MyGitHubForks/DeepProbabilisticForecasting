from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import pandas as pd


def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=False, scaler=None
):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """

    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)
    data_list = [data]
    times = df.index.values
    if add_time_in_day:
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        data_list.append(time_in_day)
    if add_day_in_week:
        day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
        day_in_week[np.arange(num_samples), :, df.index.dayofweek] = 1
        data_list.append(day_in_week)

    data = np.concatenate(data_list, axis=-1)
    # epoch_len = num_samples + min(x_offsets) - max(y_offsets)
    x, y, xTimes, yTimes = [], [], [],[]

    # t is the index of the last observation.
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]
        y_t = data[t + y_offsets, ...]
        x_time = times[t + x_offsets]
        y_time = times[t + y_offsets]
        x.append(x_t)
        y.append(y_t)
        xTimes.append(x_time)
        yTimes.append(y_time)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    xTimes = np.stack(xTimes, axis=0)
    yTimes = np.stack(yTimes, axis=0)
    return x, y, xTimes, yTimes


def generate_train_val_test(args):
    df = pd.read_hdf(args.traffic_df_filename)
    # 0 is the latest observed sample.
    x_offsets = np.sort(
        # np.concatenate(([-week_size + 1, -day_size + 1], np.arange(-11, 1, 1)))
        np.concatenate((np.arange(-11, 1, 1),))
    )
    # Predict the next one hour
    y_offsets = np.sort(np.arange(1, 13, 1))
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x, y, xTimes, yTimes = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=True,
        add_day_in_week=False,
    )

    print("x shape: ", x.shape, ", y shape: ", y.shape)
    # Write the data into npz file.
    # num_test = 6831, using the last 6831 examples as testing.
    # for the rest: 7/8 is used for training, and 1/8 is used for validation.
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train

    # train
    x_train, y_train, x_times_train, y_times_train = (
        x[:num_train],
        y[:num_train],
        xTimes[:num_train],
        yTimes[:num_train]
    )

    # val
    x_val, y_val, x_times_val, y_times_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
        xTimes[num_train:num_train + num_val],
        yTimes[num_train:num_train + num_val]
    )
    # test
    x_test, y_test, x_times_test, y_times_test = (
        x[-num_test:],
        y[-num_test:],
        xTimes[-num_test:],
        yTimes[-num_test:]
    )
    if args.down_sample:
        selection_train = list(np.random.choice(range(num_train), size=np.ceil(num_train * args.down_sample).astype(int),
            replace=False).astype(int))
        selection_val = list(np.random.choice(range(num_val), size=np.ceil(num_val * args.down_sample).astype(int),
            replace=False).astype(int))
        selection_test = list(np.random.choice(range(num_test), size=np.ceil(num_test * args.down_sample).astype(int),
            replace=False).astype(int))
        x_train = x_train[selection_train]
        y_train = y_train[selection_train]
        x_times_train = x_times_train[selection_train]
        y_times_train = y_times_train[selection_train]
        x_val = x_val[selection_val]
        y_val = y_val[selection_val]
        x_times_val = x_times_val[selection_val]
        y_times_val = y_times_val[selection_val]
        x_test = x_test[selection_test]
        y_test = y_test[selection_test]
        x_times_test = x_times_test[selection_test]
        y_times_test = y_times_test[selection_test]
    for cat in ["train", "val", "test"]:
        _x, _y, _x_times, _y_times = (
            locals()["x_" + cat],
            locals()["y_" + cat],
            locals()["x_times_"+cat],
            locals()["y_times_"+cat])
        print(cat, "x: ", _x.shape, "y:", _y.shape, "xTimes:", _x_times.shape, "yTimes:", _y_times.shape)
        if args.down_sample:
            output_dir = args.output_dir + "/down_sample_{}".format(args.down_sample)
        else:
            output_dir = args.output_dir
        if not os.path.isdir(output_dir):
                os.mkdir(output_dir)
        np.savez_compressed(
            os.path.join(output_dir, "%s.npz" % cat),
            inputs=_x,
            targets=_y,
            inputTimes=_x_times,
            targetTimes=_y_times,
        )


def main(args):
    print("Generating training data")
    generate_train_val_test(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="/users/danielzeiberg/Documents/Data/Traffic/Processed/trafficWithTime", help="Output directory."
    )
    parser.add_argument(
        "--traffic_df_filename",
        type=str,
        default="/users/danielzeiberg/Documents/Data/Traffic/df_highway_2012_4mon_sample.h5",
        help="Raw traffic readings.",
    )
    parser.add_argument(
        "--down_sample",
        type=float,
        default=0.0,
        help="fraction of each dataset to keep")
    args = parser.parse_args()
    main(args)
