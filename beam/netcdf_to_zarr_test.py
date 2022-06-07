import os
import shutil
from typing import List

from absl.testing import absltest
from ai_weather_climate.beam import netcdf_to_zarr
import numpy as np
import pandas as pd
import xarray
from xarray_beam._src import test_util
import zarr


class NetcdfToZarrTest(test_util.TestCase):

  def setUp(self):
    super().setUp()
    self.tmp_dir = os.path.join(absltest.get_default_test_tmpdir(), 'data')
    shutil.rmtree(self.tmp_dir, ignore_errors=True)
    os.makedirs(self.tmp_dir)

  def create_netcdf_files(self) -> List[str]:
    time = pd.date_range('2014-09-06', periods=12)
    lat = [0, 1, 2, 3]
    lon = [0, 1, 2, 3]
    t = xarray.Dataset(
        data_vars=dict(
            t=(['lat', 'lon', 'time'], np.random.uniform(size=(4, 4, 12)))),
        coords=dict(lat=(['lat'], lat), lon=(['lon'], lon), time=time),
        attrs=dict(description='Temperature'))
    z_list = []
    for level in [300.0, 500.0, 700.0]:
      z_level = xarray.Dataset(
          data_vars=dict(
              z=(['lat', 'lon', 'level', 'time'],
                 np.random.uniform(size=(4, 4, 1, 12)))),
          coords=dict(
              lat=(['lat'], lat),
              lon=(['lon'], lon),
              level=(['level'], [level]),
              time=time),
          attrs=dict(description='Geopotential'))
      z_list.append(z_level)

    t_file = os.path.join(self.tmp_dir, 't.nc')
    t.to_netcdf(t_file, mode='w')
    input_files = [t_file]
    for i, z_level in enumerate(z_list):
      z_file = os.path.join(self.tmp_dir, f'z_{i}.nc')
      z_level.to_netcdf(z_file, mode='w')
      input_files.append(z_file)

    return input_files

  def test_netcdf_to_zarr(self):
    input_files = self.create_netcdf_files()
    zarr_dir = os.path.join(self.tmp_dir, 'raw.zarr')
    zarr_store = zarr.storage.NestedDirectoryStore(zarr_dir)

    _ = (
        test_util.EagerPipeline()
        | netcdf_to_zarr.NetcdfToZarr(
            input_files,
            chunk_dim='time',
            chunk_size=3,
            singleton_dims=['level'],
            zarr_store=zarr_store))

    ds = xarray.open_zarr(zarr_store, consolidated=True)
    ds = ds.compute()
    ds_expected = xarray.merge(xarray.open_dataset(f) for f in input_files)

    np.testing.assert_almost_equal(ds.t.data, ds_expected.t.data)
    np.testing.assert_almost_equal(ds.z.data, ds_expected.z.data)
    self.assertTrue(ds_expected.equals(ds))


if __name__ == '__main__':
  absltest.main()
