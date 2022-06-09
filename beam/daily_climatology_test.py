from absl.testing import absltest
from ai_weather_climate.beam import daily_climatology
from ai_weather_climate.util import test_util
import numpy as np
import xarray
import xarray_beam
from xarray_beam._src import test_util as beam_test_util


class StatsUtilTest(absltest.TestCase):

  def setUp(self):
    super(StatsUtilTest, self).setUp()
    self.tmp_dir = absltest.get_default_test_tmpdir()

  def test_daily_climatology(self):
    """Test the DailyClimatology pipeline."""
    zarr_name = test_util.create_test_zarr_dataset(self.tmp_dir)
    in_ds = xarray.open_zarr(zarr_name, consolidated=True)
    in_ds_group = in_ds.groupby('time.dayofyear')

    [(_, stats)] = (
        beam_test_util.EagerPipeline()
        | daily_climatology.DailyClimatology(in_ds, time_dim='time')
        | xarray_beam.ConsolidateChunks({'dayofyear': -1}))

    np.testing.assert_array_almost_equal(
        in_ds_group.mean(dim=['time']).t.isel(level=0).compute(),
        stats.t.sel(stat='mean').isel(level=0).compute())
    np.testing.assert_array_almost_equal(
        in_ds_group.std(dim=['time']).t.isel(level=0).compute(),
        stats.t.sel(stat='std').isel(level=0).compute())


if __name__ == '__main__':
  absltest.main()
