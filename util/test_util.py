"""Utils for testing."""
import datetime
import tempfile
from typing import Tuple

import numpy as np
import xarray


def create_test_zarr_dataset(tmp_dir: str, with_nans: bool = False) -> str:
  """Create a random Zarr volume, returning the path."""
  tm_1 = [datetime.datetime(2020, 3, i) for i in range(10, 30)]
  tm_2 = [datetime.datetime(2021, 3, i) for i in range(1, 20)]
  tm = np.array(tm_1 + tm_2)
  level = np.array([300, 500, 700], dtype=np.uint32)
  lat = np.linspace(90, -90, 16, endpoint=False) - 180 / 32
  lon = np.linspace(0, 360, 16, endpoint=False) + 360 / 32
  s = tm.shape + level.shape + lat.shape + lon.shape
  t = np.random.uniform(250, 300, s)
  z = np.random.uniform(10000, 20000, s)
  if with_nans:
    t[t < 270] = np.nan
    z[z < 15000] = np.nan
  extra = np.random.uniform(0, 1, tm.shape + lat.shape + lon.shape)
  ds = xarray.Dataset(
      data_vars=dict(
          t=(['time', 'level', 'lat', 'lon'], t),
          z=(['time', 'level', 'lat', 'lon'], z),
          extra=(['time', 'lat', 'lon'], extra)),
      coords=dict(
          lon=(['lon'], lon),
          lat=(['lat'], lat),
          time=(['time'], tm),
          level=(['level'], level)),
      attrs=dict(description='Dummy weather data'))

  zarr_name = tempfile.mkdtemp(suffix='.zarr', dir=tmp_dir)
  ds.to_zarr(zarr_name, consolidated=True)
  return zarr_name


def create_test_zarr_cubesphere_dataset(
    tmp_dir: str, with_nans: bool = False) -> str:
  """Create a random Zarr volume, returning the path."""
  tm_1 = [datetime.datetime(2020, 3, i) for i in range(10, 30)]
  tm_2 = [datetime.datetime(2021, 3, i) for i in range(1, 20)]
  tm = np.array(tm_1 + tm_2)
  face_shape = (6, 48, 48)
  level = np.array([300, 500, 700], dtype=np.uint32)
  s = tm.shape + level.shape + face_shape
  t = np.random.uniform(250, 300, s)
  z = np.random.uniform(10000, 20000, s)
  if with_nans:
    t[t < 270] = np.nan
    z[z < 15000] = np.nan
  extra = np.random.uniform(0, 1, tm.shape + face_shape)
  ds = xarray.Dataset(
      data_vars=dict(
          t=(['time', 'level', 'face', 'i', 'j'], t),
          z=(['time', 'level', 'face', 'i', 'j'], z),
          extra=(['time', 'face', 'i', 'j'], extra)),
      coords=dict(time=(['time'], tm), level=(['level'], level)),
      attrs=dict(description='Dummy weather data'))

  zarr_name = tempfile.mkdtemp(suffix='.zarr', dir=tmp_dir)
  ds.to_zarr(zarr_name, consolidated=True)
  return zarr_name


def create_test_static_cubesphere_dataset(tmp_dir: str) -> str:
  """Create a random Zarr volume, returning the path."""
  tm_1 = [datetime.datetime(2020, 3, i) for i in range(10, 30)]
  tm_2 = [datetime.datetime(2021, 3, i) for i in range(1, 20)]
  tm = np.array(tm_1 + tm_2)
  face_shape = (6, 48, 48)
  days = np.arange(tm.shape[0])[np.newaxis, np.newaxis, np.newaxis, :]
  days = np.broadcast_to(days, face_shape + tm.shape)
  insolation = np.random.uniform(0, 1, face_shape + (366,))
  ds = xarray.Dataset(
      data_vars=dict(
          days=(['face', 'i', 'j', 'time'], days),
          insolation=(['face', 'i', 'j', 'dayofyear'], insolation)),
      coords=dict(time=(['time'], tm)),
      attrs=dict(description='Dummy weather data'))

  zarr_name = tempfile.mkdtemp(suffix='.zarr', dir=tmp_dir)
  ds.to_zarr(zarr_name, consolidated=True)
  return zarr_name
