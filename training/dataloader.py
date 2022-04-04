import os
import pickle
import random

import jax.numpy as jnp


class TrajectoryDataset:
    def __init__(self, trajectory_files_root, shuffle=True, online_training=False):
        if online_training:
            self.trajectories = trajectory_files_root
        else:
            # Get trajectory file names
            traj_name = [x for x in os.listdir(trajectory_files_root) if os.path.splitext(x)[1] == ".traj"]

            # Load trajectories
            self.trajectories = []
            for tn in traj_name:
                with open(os.path.join(trajectory_files_root, tn), 'rb') as f:
                    self.trajectories.extend(pickle.load(f))

        if shuffle:
            random.shuffle(self.trajectories)

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        s_0 = self.trajectories[idx][0]
        a_0 = self.trajectories[idx][1]
        reward = jnp.expand_dims(jnp.array([self.trajectories[idx][2]]), 0)
        s_1 = self.trajectories[idx][3]
        return jnp.concatenate(s_0, axis=1), jnp.stack(a_0, axis=1), reward, jnp.concatenate(s_1, axis=1)


class TrajectoryIterator:
    def __init__(self, trajectories, batch_size):
        self._trajectories = trajectories
        self.batch_size = batch_size
        self._index = 0

    def __next__(self):
        if self._index < len(self._trajectories) // self.batch_size:
            batch_data = [self._trajectories[self._index*self.batch_size + i] for i in range(self.batch_size)]
            self._index += 1
            return [jnp.concatenate([x[i] for x in batch_data], axis=0) for i in range(len(batch_data[0]))]
        raise StopIteration


class TrajectoryLoader:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __iter__(self):
        return TrajectoryIterator(self.dataset, self.batch_size)
