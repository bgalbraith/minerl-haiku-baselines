from typing import List

import minerl
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf


tf.config.experimental.set_visible_devices([], "GPU")


def load_data(environment: str, 
              batch_size: int = 32,
              n_epochs: int = 10,
              buffer_length: int = 1):
    """ Use MineRL data loader to pull samples into memory then bundle into a
    Tensorflow Dataset for managing batching behavior.
    """
    pipeline = minerl.data.make(environment)  # Assumes MINERL_DATA_ROOT is set

    trajectories = pipeline.get_trajectory_names()
    train, test = train_test_split(trajectories, test_size=1)

    train_dataset = prepare_training_dataset(pipeline, train,
                                             buffer_length=buffer_length,
                                             batch_size=batch_size,
                                             epochs=epochs)
    validation_dataset = prepare_validation_dataset(pipeline, test)

    return train_dataset, validation_dataset


def prepare_training_dataset(pipeline: minerl.data.DataPipeline,
                             trajectories: List[str],
                             buffer_length: int = 1,
                             batch_size: int = 32,
                             epochs: int = 1):
    t = 0
    povs = []
    vectors = []
    actions = []

    for trajectory in trajectories:
        for sample in pipeline.load_data(trajectory):
            povs.append(sample[0]['pov'].astype(np.float32) / 255)
            vectors.append(sample[0]['vector'])
            actions.append(sample[1]['vector'])
        t += 1
        
        if t % buffer_length == 0 or t >= len(trajectories):
            ds = tf.data.Dataset.from_tensor_slices((povs, vectors, actions))
            ds = ds.shuffle(len(povs), reshuffle_each_iteration=True)
            ds = ds.repeat(epochs)
            ds = ds.batch(batch_size)

            yield from ds.as_numpy_iterator()

            t = 0
            povs = []
            vectors = []
            actions = []


def prepare_validation_dataset(pipeline: minerl.data.DataPipeline,
                               trajectories: List[str]):
    povs = []
    vectors = []
    actions = []

    for trajectory in trajectories:
        for sample in pipeline.load_data(trajectory):
            povs.append(sample[0]['pov'].astype(np.float32) / 255)
            vectors.append(sample[0]['vector'])
            actions.append(sample[1]['vector'])
    
    return np.array(povs), np.array(vectors), np.array(actions)
