"""Dataset helpers for tests."""

import functools
from typing import Tuple

from ai_weather_climate.noaa_leap.models import dataset_util
from ai_weather_climate.noaa_leap.models import linear
import numpy as np
import tensorflow as tf
import xarray


def create_fake_gfs() -> Tuple[xarray.Dataset, tf.data.Dataset]:
  """Returns a minimal GFS dataset as an xarray Dataset and a tf Dataset."""
  ds = xarray.Dataset(
      {
          'Temperature':
              (['t', 'fc'], [[10.0, 11.0], [20.0, 22.0], [40.0, 44.0]]),
          'Cloud mixing ratio': (['t', 'fc'], np.zeros((3, 2))),
          'Relative humidity': (['t', 'fc'], np.zeros((3, 2)))
      },
      coords={
          't': [10000, 13600, 17200],
          'fc': [0, 1]
      })
  ds['Temperature'] = ds['Temperature'].expand_dims(['lat', 'lon'], [0, 1])

  ds_clean = linear.clean_up_dataset(ds)
  sample_gen = functools.partial(
      dataset_util.sample_generator, fc=1, max_nan_ratio=0.1)
  tf_ds = dataset_util.tf_dataset_from_iterable_f(
      lambda: sample_gen(ds_clean)).batch(1)
  return ds, tf_ds
