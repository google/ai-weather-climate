"""Calculate climatological statistics aggregated by day-of-year."""
import json
from typing import Iterable, Tuple

from absl import app
from absl import flags
from ai_weather_climate.util import stats_util
import apache_beam as beam
import numpy as np
import pandas
import xarray
import xarray_beam


flags.DEFINE_string(
    'input_zarr', None,
    'Path to Zarr volume')
flags.DEFINE_string('output_zarr', None, 'Path to output Zarr volume')
flags.DEFINE_string(
    'time_dim', 'time', 'Name of the time dimension')
flags.DEFINE_string(
    'dayofyear_dim', 'dayofyear', 'Name of the dayofyear dimension')
flags.DEFINE_string(
    'output_chunks',
    '{"dayofyear": 1}',
    'Output chunking scheme, expressed in json format')
flags.DEFINE_bool('skipna', True, 'Skip NaN values iff True')
flags.DEFINE_string('runner', None, 'beam.runners.Runner')

FLAGS = flags.FLAGS


def _day_of_year(t: np.datetime64) -> int:
  return pandas.to_datetime(t).timetuple().tm_yday


def _key_by_dayofyear(
    arg: Tuple[xarray_beam.Key, xarray.Dataset],
    time_dim: str
) -> Iterable[Tuple[int, xarray.Dataset]]:
  """Slice the Dataset by day of year.

  For efficiency, contiguous regions are selected with a single
  slice. Datasets that are ordered linearly by time will probably
  have better performance.

  Args:
    arg: a pair (key, ds), the key is ignored and the ds is sliced
    time_dim: the name of the time dimension to slice

  Yields:
    A sequence of (day_of_year, ds_slice) slices.
  """
  _, ds = arg
  times = ds.coords[time_dim].data
  doy = _day_of_year(times[0])
  doy_start_index = 0
  for i, t in enumerate(times):
    current_doy = _day_of_year(t)
    if current_doy != doy:
      yield doy, ds.isel({time_dim: slice(doy_start_index, i)})
      doy = current_doy
      doy_start_index = i
  yield doy, ds.isel({time_dim: slice(doy_start_index, None)})


def _compute_daily_stats(
    items: Tuple[int, Iterable[xarray.Dataset]],
    time_dim: str, dayofyear_dim: str,
    min_dayofyear: int, skipna: bool
) -> Tuple[xarray_beam.Key, xarray.Dataset]:
  """Compute the daily stats.

  Args:
    items: (day_of_year, datasets) containing all the datasets for the
      day_of_year
    time_dim: the name of the time dimension
    dayofyear_dim: name of the day-of-year dimension
    min_dayofyear: the minimum day-of-year over all inputs
    skipna: ignore NaNs iff True

  Returns:
    A Dataset containing the statistics for that day of year.
  """
  day_of_year, chunks = items
  ds = xarray.concat(list(chunks), time_dim)
  ds_stats = stats_util.compute_stats(ds, {time_dim}, skipna)

  # Add a dayofyear dimension.
  ds_stats = ds_stats.expand_dims({dayofyear_dim: [day_of_year]})

  # Construct the proper chunk key. The offsets dict maps to index, so we have
  # to subtract the min_dayofyear to get the index.  Note that Jan 1 is day 1.
  offsets = {dim: 0 for dim in ds_stats.dims}
  offsets[dayofyear_dim] = day_of_year - min_dayofyear
  key = xarray_beam.Key(offsets=offsets)

  return key, ds_stats


class DailyClimatology(beam.PTransform):
  """Calculate the stats for a Zarr file."""

  def __init__(self,
               in_ds: xarray.Dataset,
               time_dim: str = 'time',
               dayofyear_dim: str = 'dayofyear',
               time_chunk_size: int = 1,
               skipna: bool = True):
    """Calculate stats for a Zarr file.

    Args:
      in_ds: the Zarr dataset
      time_dim: name of the time dimension (default 'time')
      dayofyear_dim: name of the day-of-year dimension (default 'dayofyear')
      time_chunk_size: chunk size for the time_dim (default 1)
      skipna: ignore NaNs iff skipna is True
    """
    self.in_ds = in_ds
    self.in_chunks = {dim: time_chunk_size if dim == time_dim else -1
                      for dim in in_ds.dims}
    self.time_dim = time_dim
    self.dayofyear_dim = dayofyear_dim
    self.skipna = skipna
    times = pandas.to_datetime(in_ds.coords[time_dim].data)
    days = {t.timetuple().tm_yday for t in times}
    assert max(days) - min(days) + 1 == len(days), (
        f'Sparse {dayofyear_dim} ranges are not supported: {days}')
    self.min_dayofyear = min(days)

  # Compute the daily climatology by splitting the input into chunks,
  # slicing by dayofyear, joining into a single xarray, and
  # computing stats.
  #
  # The join is probably the most expensive operation. If this becomes a
  # bottleneck, an alternative is to calculate stats for each slice separately,
  # then use stats_util.reduce_stats to combine the results.
  #
  # Also note that flox may provide an efficient groupby operation that might
  # simplify this pipeline. https://flox.readthedocs.io/en/latest/
  def expand(self, p: beam.PTransform) -> beam.PTransform:
    return (
        p
        | xarray_beam.DatasetToChunks(
            self.in_ds, self.in_chunks, num_threads=32)
        | beam.FlatMap(_key_by_dayofyear, time_dim=self.time_dim)
        | beam.GroupByKey()
        | beam.Map(
            _compute_daily_stats,
            min_dayofyear=self.min_dayofyear,
            time_dim=self.time_dim, dayofyear_dim=self.dayofyear_dim,
            skipna=self.skipna))


def main(argv):
  if not FLAGS.input_zarr:
    raise ValueError('--input_zarr option must be specified')
  if not FLAGS.output_zarr:
    raise ValueError('--output_zarr option must be specified')

  # Use the default chunk size.
  in_ds = xarray.open_zarr(FLAGS.input_zarr, consolidated=True)
  in_chunks = in_ds.chunks
  time_chunk_size = in_chunks[FLAGS.time_dim] if in_chunks else 1

  # Reopen the dataset without chunks for efficiency.
  in_ds = xarray.open_zarr(FLAGS.input_zarr, chunks=None, consolidated=True)
  out_chunks = json.loads(FLAGS.output_chunks)
  for dim in set(in_ds.dims) - {FLAGS.time_dim} - set(out_chunks):
    out_chunks[dim] = -1
  out_chunks['stat'] = -1

  with  beam.Pipeline(runner=FLAGS.runner, argv=argv) as p:
    _ = (
        p
        | DailyClimatology(
            in_ds, time_dim=FLAGS.time_dim, time_chunk_size=time_chunk_size,
            dayofyear_dim=FLAGS.dayofyear_dim, skipna=FLAGS.skipna)
        | xarray_beam.ConsolidateChunks(out_chunks)
        | xarray_beam.ChunksToZarr(FLAGS.output_zarr))


if __name__ == '__main__':
  app.run(main)
