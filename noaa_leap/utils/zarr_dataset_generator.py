"""Creates a tf.data.Dataset from a Zarr array store.

Assumes the Zarr data has been converted into a model ready format containing
both an input feature and target label data variable and has NaNs that have
been imputed.
"""

import dataclasses
import multiprocessing
import random
from typing import Mapping, Optional, Tuple

import tensorflow as tf
import xarray


@dataclasses.dataclass(eq=False)
class DatasetReadConfig:
  """Configures input reading pipeline for tf.data.Dataset.

  Attributes:
    options: `tf.data.Options()`, dataset options to use. Note that when
      `shuffle_files` is True and no seed is defined, experimental_deterministic
      will be set to False internally, unless it is defined here.
    shuffle: `bool`, whether to randomize the slices for the dataset generator
      and forwarded to `tf.data.Dataset` to shuffle batched data elements.
    shuffle_seed: `int`, seed forwarded to `random` and during slice shuffling.
    interleave_cycle_length: `int`, forwarded to `tf.data.Dataset.interleave`.
    interleave_block_length: `int`, forwarded to `tf.data.Dataset.interleave`.
    interleave_deterministic: `bool`, controls the order in which the interleave
      produces elements.
    num_parallel_calls_for_interleave: Optional[int] =
        tf.data.experimental.AUTOTUNE : The number of parallel calls for
          interleaving files. By default using tf.data's AUTOTUNE.
    num_shards: `int`, number of shards to split dataset. By default, it uses
      `number_of_processors` // 4 to balance parallelism and memory overhead
      from Dask.
  """
  # General tf.data.Dataset parameters.
  options: tf.data.Options = dataclasses.field(default_factory=tf.data.Options)
  # shuffle parameters
  shuffle: bool = True
  shuffle_seed: Optional[int] = None
  # Number of shards to split the dataset by.
  num_shards: Optional[int] = multiprocessing.cpu_count() // 4
  # Interleave parameters:
  #   Setting interleave_deterministic to False helps increase parallelism and
  #   setting interleave_cycle_length and num_parallel_calls_for_interleave
  #   to num_shards with interleave_block_length set to 1 provides the best
  #   performance. num_shards should be set to a value <<
  #   multiprocessing.cpu_count().
  interleave_deterministic: Optional[bool] = False
  interleave_cycle_length: Optional[int] = 1
  interleave_block_length: Optional[int] = 1
  num_parallel_calls_for_interleave: Optional[int] = multiprocessing.cpu_count(
  ) // 4


@dataclasses.dataclass(eq=False)
class ZarrConfig:
  """Configures zarr dataset info.

  Attributes:
    chunks: `dict`, Chunk sizes along each dimension, e.g., {'x': 5, 'y': 5} or
      None. By default, it uses `None` to prevent contention in multi-threading
      between Dask and Tensorflow.
    consolidated: `bool`, whether to open the store using zarr’s consolidated
      metadata capability. Only works for stores that have already been
      consolidated.
    x_key: `str`, the key name for the X data variable.
    y_key: `str`, the key name for the Y data variable.
    time_key: `str`, the key name for the time coordinate.
  """
  chunks: Optional[Mapping[str, int]] = None
  consolidated: bool = True
  x_key: Optional[str] = None
  y_key: Optional[str] = None
  time_key: Optional[str] = None


class ZarrDataSetGenerator:
  """tf.data.Dataset generator for a pre-processed zarr dataset."""

  def __init__(self,
               root,
               batch_size,
               zarr_config: ZarrConfig,
               read_config: Optional[DatasetReadConfig] = None):
    self.root = root
    self.batch_size = batch_size
    self.zarr_config = zarr_config
    self.read_config = (read_config if read_config is not None
                        else DatasetReadConfig())

  def _dataset_generator(self, start_shard_index, shard_size):
    """Generates a slice of data."""
    dataset = xarray.open_zarr(
        self.root,
        consolidated=self.zarr_config.consolidated,
        chunks=self.zarr_config.chunks)

    end_shard_index = start_shard_index + shard_size
    range_list = list(
        range(start_shard_index, end_shard_index, self.batch_size))
    slices = list(zip(range_list[:-1], range_list[1:]))

    if self.read_config.shuffle:
      random.shuffle(slices)

    for index_slice in slices:
      sliced_dataset = dataset.isel(
          {self.zarr_config.time_key: slice(index_slice[0], index_slice[1])})
      input_features = sliced_dataset[self.zarr_config.x_key]
      label = sliced_dataset[self.zarr_config.y_key]

      yield input_features, label

  def as_dataset(
      self,
      repeat: bool = False
  ) -> Tuple[tf.data.Dataset, Tuple[int, int, int, int, int]]:
    """Create a tf.data.Dataset from pre-processed Zarr data."""
    random.seed(self.read_config.shuffle_seed)

    ds = xarray.open_zarr(
        self.root,
        consolidated=self.zarr_config.consolidated,
        chunks=self.zarr_config.chunks)

    num_samples, height, width, num_input_channels = ds[
        self.zarr_config.x_key].shape
    num_output_channels = ds[self.zarr_config.y_key].shape[-1]

    shard_size = num_samples // self.read_config.num_shards
    shard_indices = list(range(0, num_samples, shard_size))
    dataset = tf.data.Dataset.from_tensor_slices(shard_indices)

    dataset = dataset.interleave(
        lambda x: tf.data.Dataset.from_generator(  # pylint: disable=g-long-lambda
            self._dataset_generator,
            output_signature=(
                tf.TensorSpec(
                    shape=(self.batch_size, height, width, num_input_channels),
                    dtype=tf.float32),
                tf.TensorSpec(
                    shape=(self.batch_size, height, width, num_output_channels),
                    dtype=tf.float32)
            ),
            args=(x, shard_size)),
        num_parallel_calls=self.read_config.num_parallel_calls_for_interleave,
        cycle_length=self.read_config.interleave_cycle_length,
        block_length=self.read_config.interleave_block_length,
        deterministic=self.read_config.interleave_deterministic)

    if self.read_config.shuffle:
      dataset = dataset.shuffle(self.batch_size)

    if repeat:
      dataset = dataset.repeat()

    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.with_options(self.read_config.options)

    return dataset, (height, width, num_input_channels, num_output_channels,
                     num_samples)
