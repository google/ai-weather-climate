"""Utilities for generating (forecast, error) -> error samples from GFS."""

# pylint: disable=g-doc-args

from typing import Any, Dict, Callable, Iterable, Sequence

import numpy as np
import tensorflow as tf
import xarray


def to_stacked_numpy(datasets: Sequence[xarray.Dataset]) -> np.ndarray:
  """Stacks all data in datasets into a single numpy array."""
  arrays = []
  for ds in datasets:
    ds = ds.compute()
    for v in ds.data_vars:
      d = ds[v].data
      assert d.ndim in [2, 3], f'Expected 2 or 3 dims, got {d.ndim} for {v}'
      if d.ndim == 2:
        d = np.expand_dims(d, axis=-1)
      arrays.append(d)

  return np.concatenate(arrays, axis=-1)


def to_datasets(
    data: np.ndarray,
    template_datasets: Sequence[xarray.Dataset]) -> Sequence[xarray.Dataset]:
  """Creates datasets from a single numpy array."""
  out = []
  i = 0
  assert data.ndim == 3, f'Expected 3 dims, got {data.ndim}'
  for ds in template_datasets:
    ds_copy = ds.copy()
    for v in ds_copy.data_vars:
      if ds_copy[v].ndim == 2:
        ds_copy[v].data = data[:, :, i]
        i += 1
      else:
        assert ds_copy[v].ndim == 3, (f'Expected 3 dims, got {ds_copy[v].ndim} '
                                      f'for {v}')
        ds_copy[v].data = data[:, :, i:i + ds_copy[v].shape[2]]
        i += ds_copy[v].shape[2]
    out.append(ds_copy)
  assert i == data.shape[2], (
      f'Data has {data.shape[2]} layers, templates used {i}')
  return out


def get_sample(ds: xarray.Dataset, t: xarray.DataArray,
               fc: int) -> Dict[str, xarray.Dataset]:
  """Returns sample data at time t for forecast time fc."""
  fc_s = fc * 3600
  try:
    forecast = ds.sel(t=t - fc_s, fc=fc)
    analysis = ds.sel(t=t - fc_s, fc=0)
    last_forecast = ds.sel(t=t - 2 * fc_s, fc=fc)
    future_analysis = ds.sel(t=t, fc=0)
  except KeyError:
    return {}
  last_error = analysis - last_forecast
  error = future_analysis - forecast
  return {'forecast': forecast, 'last_error': last_error, 'error': error}


def nan_ratio(x: np.ndarray) -> np.ndarray:
  """Returns the ratio of nans at each coordinate of the 3rd dimension."""
  assert x.ndim == 3, f'Expected 3 dims, got {x.ndim}'
  return np.count_nonzero(np.isnan(x), axis=(0, 1)) / (x.size / x.shape[-1])


def fill_nans_with_mean(x: np.ndarray) -> np.ndarray:
  """Replaces each nan at (x, y, z) with mean of all values at that z."""
  assert x.ndim == 3, f'Expected 3 dims, got {x.ndim}'
  return np.nan_to_num(x, nan=np.nanmean(x, axis=(0, 1)), copy=False)


def convert_to_nan_free_numpy(datasets: Dict[str, xarray.Dataset],
                              max_nan_ratio: float) -> Dict[str, np.ndarray]:
  """Converts values in datasets to numpy array, checking / filling nans."""
  out = {}
  for name, data in datasets.items():
    data_array = to_stacked_numpy([data])
    if max(nan_ratio(data_array)) > max_nan_ratio:
      return {}
    out[name] = fill_nans_with_mean(data_array)
  return out


def sample_generator(ds: xarray.Dataset,
                     fc: int,
                     max_nan_ratio: float = 0.1,
                     shuffle_seed: int = 0) -> Iterable[Dict[str, np.ndarray]]:
  """Yields pairs of (inputs, label) from ds for forecast time fc.

  Iterates through all timesteps in dimension 't'. Skips timesteps if the
  number of nans in any variable/level is greater than max_nan_ratio. Otherwise
  fills nans with the average of the level.
  """
  if shuffle_seed != 0:
    rs = np.random.RandomState(shuffle_seed)
    times = ds['t'].copy().data
    rs.shuffle(times)
  else:
    times = ds['t'].data
  for t in times:
    sample = get_sample(ds, t, fc)
    if not sample:
      continue
    out = convert_to_nan_free_numpy(sample, max_nan_ratio)
    if not out:
      continue
    yield out


def tensor_spec_from_numpy_array(x: np.ndarray) -> tf.TensorSpec:
  return tf.TensorSpec(shape=x.shape, dtype=tf.dtypes.as_dtype(x.dtype))


def tf_dataset_from_iterable_f(
    iter_f: Callable[[], Iterable[Any]]) -> tf.data.Dataset:
  sample = next(iter(iter_f()))
  return tf.data.Dataset.from_generator(
      iter_f,
      output_signature=tf.nest.map_structure(tensor_spec_from_numpy_array,
                                             sample))
