from absl.testing import absltest

from ai_weather_climate.noaa_leap.models import dataset_util
import numpy as np
import xarray


class DatasetUtilTest(absltest.TestCase):

  def test_to_stacked_numpy(self):
    a_2d = 2 * np.ones((2, 3))
    a_3d = 3 * np.ones((2, 3, 4))
    ds_2d = xarray.Dataset({'a': (['x', 'y'], a_2d)})
    ds_3d = xarray.Dataset({'a': (['x', 'y', 'z'], a_3d)})

    expected = np.concatenate([np.expand_dims(a_2d, axis=-1), a_3d], axis=-1)
    np.testing.assert_allclose(
        dataset_util.to_stacked_numpy([ds_2d, ds_3d]), expected)

  def test_to_datasets(self):
    a_2d = 2 * np.ones((2, 3))
    a_3d = 3 * np.ones((2, 3, 4))

    ds_2d = xarray.Dataset({'a': (['x', 'y'], a_2d)})
    ds_3d = xarray.Dataset({'a': (['x', 'y', 'z'], a_3d)})

    template = [xarray.zeros_like(ds_2d), xarray.zeros_like(ds_3d)]
    actual = dataset_util.to_datasets(
        np.concatenate([np.expand_dims(a_2d, axis=-1), a_3d], axis=-1),
        template)
    self.assertLen(actual, 2)
    xarray.testing.assert_allclose(actual[0], ds_2d)
    xarray.testing.assert_allclose(actual[1], ds_3d)

  def test_get_sample(self):
    ds = xarray.Dataset(
        {'a': (['t', 'fc'], [[10.0, 11.0], [20.0, 22.0], [40.0, 44.0]])},
        coords={
            't': [10000, 13600, 17200],
            'fc': [0, 1]
        })
    with self.subTest('normal'):
      actual = dataset_util.get_sample(ds, 17200, 1)
      self.assertIsNotNone(actual)
      expected_forecast = ds.sel(t=13600, fc=1)
      expected_last_error = ds.sel(t=13600, fc=0) - ds.sel(t=10000, fc=1)
      expected_error = ds.sel(t=17200, fc=0) - ds.sel(t=13600, fc=1)
      xarray.testing.assert_allclose(actual['forecast'], expected_forecast)
      xarray.testing.assert_allclose(actual['last_error'], expected_last_error)
      xarray.testing.assert_allclose(actual['error'], expected_error)

    with self.subTest('missing_data'):
      actual = dataset_util.get_sample(ds, 13600, 1)
      self.assertFalse(actual)

  def test_get_sample_generator(self):
    ds = xarray.Dataset(
        {'a': (['t', 'fc'], [[10.0, 11.0], [20.0, 22.0], [40.0, 44.0]])},
        coords={
            't': [10000, 13600, 17200],
            'fc': [0, 1]
        })
    ds['a'] = ds['a'].expand_dims(['lat', 'lon'], [0, 1])

    with self.subTest('normal'):
      samples = list(dataset_util.sample_generator(ds, 1))
      self.assertLen(samples, 1)
      np.testing.assert_allclose(samples[0]['forecast'], [[[22.0]]])
      np.testing.assert_allclose(samples[0]['last_error'], [[[20.0 - 11.0]]])
      np.testing.assert_allclose(samples[0]['error'], [[[40.0 - 22.0]]])

    with self.subTest('with_nans'):
      ds_with_nan = ds.copy(ds)
      ds_with_nan['a'].loc[dict(t=13600, fc=1)] = np.nan
      ds_with_nans = xarray.concat([ds, ds_with_nan], 'lat')
      self.assertEmpty(
          list(
              dataset_util.sample_generator(ds_with_nans, 1,
                                            max_nan_ratio=0.4)))

      samples = list(
          dataset_util.sample_generator(ds_with_nans, 1, max_nan_ratio=0.5))
      self.assertLen(samples, 1)
      # The missing value is filled with the mean but we have only one other
      # value, so the fill value is the same as that.
      np.testing.assert_allclose(samples[0]['forecast'], [[[22.0]], [[22.0]]])
      np.testing.assert_allclose(samples[0]['last_error'],
                                 [[[20.0 - 11.0]], [[20.0 - 11.0]]])
      np.testing.assert_allclose(samples[0]['error'],
                                 [[[40.0 - 22.0]], [[40.0 - 22.0]]])

  def test_get_sample_generator_shuffled(self):
    ds = xarray.Dataset(
        {
            'a': (['t', 'fc'], [[10.0, 11.0], [20.0, 22.0], [40.0, 44.0],
                                [80.0, 88.0]])
        },
        coords={
            't': [10000, 13600, 17200, 20800],
            'fc': [0, 1]
        })
    ds['a'] = ds['a'].expand_dims(['lat', 'lon'], [0, 1])

    # No shuffle.
    samples = list(dataset_util.sample_generator(ds, 1, shuffle_seed=0))
    self.assertLen(samples, 2)
    np.testing.assert_allclose(samples[0]['forecast'], [[[22.0]]])

    # With shuffle.
    samples = list(dataset_util.sample_generator(ds, 1, shuffle_seed=1))
    self.assertLen(samples, 2)
    # Just testing that the first element is not the first one in the
    # dataset. This won't hold for any arbitrary seed, but it is true for this
    # particular one.
    np.testing.assert_allclose(samples[0]['forecast'], [[[44.0]]])

  def test_tf_dataset_from_iterable_f(self):
    iter_f = lambda: [i * np.ones((4, 3)) for i in range(2)]
    ds = dataset_util.tf_dataset_from_iterable_f(iter_f)
    actual = list(ds.as_numpy_iterator())
    self.assertLen(actual, 2)
    np.testing.assert_allclose(actual[0], np.zeros((4, 3)))
    np.testing.assert_allclose(actual[1], np.ones((4, 3)))


if __name__ == '__main__':
  absltest.main()
