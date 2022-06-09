"""Stats computations."""
from typing import Dict, Set, Text

import numpy as np
import xarray

# Stat names.
STATS = ['mean', 'std', 'max', 'min', 'count', 'isnull', 'p25', 'p50', 'p75']


def _scalar_stat(ds: xarray.Dataset, stat_name: str) -> xarray.Dataset:
  ds = ds.expand_dims(dim='stat', axis=0)
  ds.coords['stat'] = [stat_name]
  return ds


def _concat_stats(scalar_stats: Dict[Text, xarray.Dataset]) -> xarray.Dataset:
  # The consolidated metadata is written separately, in STATS order,
  # so ensure that the stats are in the same order.
  stats = [_scalar_stat(scalar_stats[name], name) for name in STATS]
  return xarray.concat(stats, dim='stat')


def _reduce_std(ds: xarray.Dataset, dim: Text) -> xarray.Dataset:
  """Calculate standard deviation across the specified dimension."""
  # Uses Chan's algorithm,
  # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm.
  # NOTE: this algorithm is iterative. We may want to investigate whether there
  # is a more direct way to compute.
  ds_a = ds.isel({dim: 0})
  n_a = ds_a.sel(stat='count')
  mean_a = ds_a.sel(stat='mean')
  m2_a = ds_a.sel(stat='std')**2 * n_a
  for i in range(1, ds.dims[dim]):
    ds_b = ds.isel({dim: i})
    n_b = ds_b.sel(stat='count')
    mean_b = ds_b.sel(stat='mean')
    m2_b = ds_b.sel(stat='std')**2 * n_b
    n_ab = n_a + n_b
    mean_ab = (n_a * mean_a + n_b * mean_b) / n_ab
    m2_ab = m2_a + m2_b + (mean_b - mean_a)**2 * n_a * n_b / n_ab
    n_a = n_ab
    mean_a = mean_ab
    m2_a = m2_ab
  return np.sqrt(m2_a / n_a)


def reduce_stats(ds: xarray.Dataset, dim: Text) -> xarray.Dataset:
  """Reduce the stats along one of the dimensions.

  For example, to get accumulated statistics across time,
  use reduce(stats, dim='t').

  NOTE: reduced quantiles are calculated as medians of individual quantiles,
  which is not accurate.

  Args:
    ds: a Dataset containing stats
    dim: the dimension to reduce

  Returns:
    A Dataset containing aggregated stats across the dimension.
  """
  count = ds.sel(stat='count')
  n = count.sum(dim=dim)
  stats = {
      'mean': (ds.sel(stat='mean') * count).sum(dim=dim) / n,
      # Note: standard deviation is difficult to compute in one line. Methods
      # like ds.sel(stat='mean').std(dim=dim) or ds.sel(stat='std').std(dim=dim)
      # would compute standard deviation over *stats*, which is not correct.
      # For example, if each element along dim is identical, std of stats would
      # return 0, which is not correct.  Call an incremental procedure for
      # calculating std (time complexity is O(|dim|)).
      'std': _reduce_std(ds, dim),
      'max': ds.sel(stat='max').max(dim=dim),
      'min': ds.sel(stat='min').min(dim=dim),
      'count': n,
      'isnull': ds.sel(stat='isnull').sum(dim=dim),
      # TODO(jyh): For quantiles, we take medians of the individual quantiles.
      # Replace with some incremental method for computing approximate
      # quantiles. Possibly https://doi.org/10.1145/347090.347195 may be a place
      # to start.
      'p25': ds.sel(stat='p25').quantile(0.5, dim=dim),
      'p50': ds.sel(stat='p50').quantile(0.5, dim=dim),
      'p75': ds.sel(stat='p75').quantile(0.5, dim=dim),
  }
  return _concat_stats(stats)


def compute_stats(ds: xarray.Dataset,
                  reduce_dims: Set[Text],
                  skipna: bool = True) -> xarray.Dataset:
  """Compute the stats for a dataset.

  The stats are placed along a new 'stat' dimension, including:
    'mean': arithmetic mean
    'std': standard deviation
    'max': max value
    'min': min value
    'count': number of non-null values
    'isnull': number of null values
    'p25': 25th percentile
    'p50': median
    'p75': 75th percentile

  Args:
    ds: the dataset
    reduce_dims: dimensions to compute stats over
    skipna: ignore NaNs iff skipna is True

  Returns:
    A Dataset with the stats.
  """
  scalar_stats = {
      'mean': ds.mean(dim=reduce_dims, skipna=skipna),
      'std': ds.std(dim=reduce_dims, skipna=skipna),
      'max': ds.max(dim=reduce_dims, skipna=skipna),
      'min': ds.min(dim=reduce_dims, skipna=skipna),
      'count': ds.notnull().sum(dim=reduce_dims),
      'isnull': ds.isnull().sum(dim=reduce_dims),
  }
  ds_quantile = ds.quantile([.25, .5, .75], dim=reduce_dims, skipna=skipna)
  ds_quantile = ds_quantile.rename(quantile='stat')
  scalar_stats['p25'] = ds_quantile.isel(stat=0)
  scalar_stats['p50'] = ds_quantile.isel(stat=1)
  scalar_stats['p75'] = ds_quantile.isel(stat=2)
  return _concat_stats(scalar_stats)
