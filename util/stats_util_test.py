from absl.testing import absltest
from ai_weather_climate.util import test_util
from ai_weather_climate.util import stats_util
import numpy as np
import xarray


class StatsUtilTest(absltest.TestCase):

  def setUp(self):
    super(StatsUtilTest, self).setUp()
    self.tmp_dir = absltest.get_default_test_tmpdir()

  def test_compute_stats(self):
    """Test the compute_stats function."""
    zarr_name = test_util.create_test_zarr_dataset(self.tmp_dir)
    ds = xarray.open_zarr(zarr_name, consolidated=True)

    reduce_dims = ['lat', 'lon']
    stats = stats_util.compute_stats(ds, reduce_dims=reduce_dims)
    np.testing.assert_array_almost_equal(
        ds.t.mean(dim=reduce_dims).isel(time=0, level=0).compute(),
        stats.t.sel(stat='mean').isel(time=0, level=0).compute())
    np.testing.assert_array_almost_equal(
        ds.t.std(dim=reduce_dims).isel(time=0, level=0).compute(),
        stats.t.sel(stat='std').isel(time=0, level=0).compute())

  def test_reduce_stats(self):
    """Test the reduce_stats function."""
    zarr_name = test_util.create_test_zarr_dataset(self.tmp_dir)
    ds = xarray.open_zarr(zarr_name, consolidated=True)

    reduce_dims = ['lat', 'lon']
    stats = stats_util.compute_stats(ds, reduce_dims=reduce_dims)
    stats = stats.chunk({'time': -1})
    stats = stats_util.reduce_stats(stats, dim='time')
    np.testing.assert_array_almost_equal(
        ds.t.mean(dim=['time'] + reduce_dims).isel(level=0).compute(),
        stats.t.sel(stat='mean').isel(level=0).compute())
    np.testing.assert_array_almost_equal(
        ds.t.std(dim=['time'] + reduce_dims).isel(level=0).compute(),
        stats.t.sel(stat='std').isel(level=0).compute())


if __name__ == '__main__':
  absltest.main()
