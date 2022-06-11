r"""Create TFRecord files containing train, test, and validation TFExamples."""
from typing import Union

from absl import app
from absl import flags
import apache_beam as beam
import numpy as np
import pandas
import tensorflow as tf
import xarray


flags.DEFINE_string(
    'input_zarr', None, 'Path to dataset containing the input')
flags.DEFINE_string(
    'stats_zarr', None, 'Path to dataset containing statistics')
flags.DEFINE_string(
    'static_zarr', None, 'Path to dataset containing static values')
flags.DEFINE_string('output_rio', None, 'Output path for recordio files')

# The train/test/validate splits are contiguous sequences partitioned by time.
#
# TODO(jyh): implement other kinds of splits, for example partition each month
# or each week.
flags.DEFINE_string(
    'train_date_range', '1970-01-01:2000-01-01',
    'Date range for training data [lower bound is closed, '
    'upper is open)')
flags.DEFINE_string(
    'validate_date_range', '2000-01-01:2010-01-01',
    'Date range for validation dataset [lower bound is closed, '
    'upper is open)')
flags.DEFINE_string(
    'test_date_range', '2010-01-01:',
    'Date range for evaluation dataset [lower bound is closed, '
    'upper is open)')
flags.DEFINE_string('runner', None, 'beam.runners.Runner')

FLAGS = flags.FLAGS


def _parse_date_range(s: str) -> Union[slice, str]:
  dl = [x if x else None for x in s.split(':')]
  if not dl:
    return slice(None, None)
  if len(dl) == 1:
    return s
  return slice(*dl)


class MakeExamples(beam.PTransform):
  """Convert the raw dataset to examples."""

  def __init__(
      self,
      in_ds: xarray.Dataset,
      stats_ds: xarray.Dataset,
      static_ds: xarray.Dataset,
      epsilon: float = 1e-6):
    """Calculate stats for a Zarr file.

    Args:
      in_ds: the Zarr dataset
      stats_ds: climatology for in_ds
      static_ds: additional static inputs
      epsilon: small number to prevent division by zero
    """
    self.in_ds = in_ds
    self.stats_ds = stats_ds
    self.static_ds = static_ds
    self.epsilon = epsilon

  def make_sample(self, t: np.datetime64) -> xarray.DataArray:
    """Create a single DataArray for the sample."""
    dayofyear = pandas.to_datetime(t).timetuple().tm_yday
    sample = self.in_ds.sel(time=t, drop=True)
    stats = self.stats_ds.sel(dayofyear=dayofyear, drop=True)
    sample = (sample - stats.sel(stat='mean'))
    sample = sample / (stats.sel(stat='std') + self.epsilon)
    static = self.static_ds.sel(drop=True, time=t, dayofyear=dayofyear)
    channels = xarray.merge((sample, static))
    channels = channels.to_stacked_array(
        new_dim='channel', sample_dims=['face', 'i', 'j'])
    channels = channels.transpose('face', 'i', 'j', 'channel')
    return channels.compute()

  def make_example(self, t: np.datetime64) -> bytes:
    """Create the tensorflow example."""
    # Get the input sample.
    in_img = self.make_sample(t)

    # In this version, we train for the identity, where the output is the same
    # as the input.
    #
    # TODO(jyh): implement other kinds of examples.
    out_img = in_img

    # Construct and return the example.
    example = tf.train.Example(
        features=tf.train.Features(feature={
            'key': tf.train.Feature(
                int64_list=tf.train.Int64List(
                    value=[t.astype('datetime64[s]').astype(np.int64)])),
            'input': tf.train.Feature(
                float_list=tf.train.FloatList(value=in_img.data.ravel())),
            'output': tf.train.Feature(
                float_list=tf.train.FloatList(value=out_img.data.ravel())),
        }))
    return example.SerializeToString()

  def expand(self, p: beam.PTransform) -> beam.PTransform:
    times = list(self.in_ds.time.data)
    return (
        p
        | beam.Create(times)
        | beam.Reshuffle()
        | beam.Map(self.make_example))


def main(argv):
  if not FLAGS.input_zarr:
    raise ValueError('--input_zarr option must be specified')
  if not FLAGS.stats_zarr:
    raise ValueError('--stats_zarr option must be specified')
  if not FLAGS.static_zarr:
    raise ValueError('--static_zarr option must be specified')
  if not FLAGS.output_rio:
    raise ValueError('--output_rio option must be specified')

  with  beam.Pipeline(runner=FLAGS.runner, argv=argv) as p:
    # Open the datasets.
    in_ds = xarray.open_zarr(
        FLAGS.input_zarr, chunks=None, consolidated=True)
    stats_ds = xarray.open_zarr(
        FLAGS.stats_zarr, chunks=None, consolidated=True)
    static_ds = xarray.open_zarr(
        FLAGS.static_zarr, chunks=None, consolidated=True)

    date_ranges = {
        'train': _parse_date_range(FLAGS.train_date_range),
        'test': _parse_date_range(FLAGS.test_date_range),
        'validate': _parse_date_range(FLAGS.validate_date_range),
    }

    for mode in ['train', 'validate', 'test']:
      mode_ds = in_ds.sel(time=date_ranges[mode])
      _ = (
          p
          | f'{mode}-make-examples' >> MakeExamples(
              mode_ds, stats_ds, static_ds)
          | f'{mode}-write-tfrecord' >> beam.io.WriteToTFRecord(
              f'{FLAGS.output_rio}-{mode}',
              file_name_suffix='.rio'))


if __name__ == '__main__':
  app.run(main)
