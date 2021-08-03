"""Tests for zarr_dataset_generator."""

import os
import shutil
import tempfile

from absl import flags
from absl.testing import absltest
from ai_weather_climate.noaa_leap.utils import zarr_dataset_generator as zdg
import numpy as np
import xarray as xr

FLAGS = flags.FLAGS


class ZarrDatasetGeneratorTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.tmpdir = tempfile.mkdtemp(dir=absltest.get_default_test_tmpdir())

  def tearDown(self):
    super().tearDown()
    shutil.rmtree(self.tmpdir)

  def copy_testdata_to_tmp(self, name, size):
    x_data = np.ones((16, 32, 8))
    y_data = np.ones((16, 32, 4))
    # Create a dataset of np.ones(...) * k to create a uniform dataset for
    # testing purposes.
    ds = xr.Dataset(
        {
            'x': (('t', 'i', 'j', 'z_input'),
                  np.stack([k * x_data for k in range(1, size + 1)])),
            'y': (('t', 'i', 'j', 'z_output'),
                  np.stack([k * y_data for k in range(1, size + 1)]))
        },
        coords={
            't': list(range(size)),
        },
    )

    tmp_path = os.path.join(self.tmpdir, name)
    ds.to_zarr(tmp_path)

    return tmp_path

  def test_get_batch_dataset(self):
    zarr_root = self.copy_testdata_to_tmp('test.zarr', size=4)
    zarr_config = zdg.ZarrConfig(
        x_key='x', y_key='y', time_key='t', consolidated=False)
    read_config = zdg.DatasetReadConfig(
        num_shards=1,
        interleave_cycle_length=1,
        num_parallel_calls_for_interleave=1)
    generator = zdg.ZarrDataSetGenerator(
        root=zarr_root,
        batch_size=2,
        zarr_config=zarr_config,
        read_config=read_config)
    ds, stats = generator.as_dataset()
    height, width, num_in_channels, num_out_channels, num_samples = stats

    ds = ds.take(1)
    l = list(ds.as_numpy_iterator())
    batched_examples = l[0]
    x = batched_examples[0]
    y = batched_examples[1]
    assert x.shape == (2, height, width, num_in_channels)
    assert y.shape == (2, height, width, num_out_channels)
    assert num_samples == 4

  def test_no_shuffle_dataset(self):
    zarr_root = self.copy_testdata_to_tmp('test.zarr', size=4)
    zarr_config = zdg.ZarrConfig(
        x_key='x', y_key='y', time_key='t', consolidated=False)
    read_config = zdg.DatasetReadConfig(
        shuffle=False,
        num_shards=1,
        interleave_cycle_length=1,
        num_parallel_calls_for_interleave=1,
        interleave_deterministic=True)
    generator = zdg.ZarrDataSetGenerator(
        root=zarr_root,
        batch_size=1,
        zarr_config=zarr_config,
        read_config=read_config)
    ds, stats = generator.as_dataset()
    height, width, num_in_channels, _, _ = stats

    ds = ds.take(1)
    l = list(ds.as_numpy_iterator())
    batched_examples = l[0]
    x = batched_examples[0]
    # Both arrays should be equal to np.ones(...).
    np.testing.assert_allclose(x, np.ones((1, height, width, num_in_channels)))

  def test_shuffle_dataset(self):
    zarr_root = self.copy_testdata_to_tmp('test.zarr', size=4)
    zarr_config = zdg.ZarrConfig(
        x_key='x', y_key='y', time_key='t', consolidated=False)
    read_config = zdg.DatasetReadConfig(
        shuffle=True,
        shuffle_seed=1,
        num_shards=1,
        interleave_cycle_length=1,
        num_parallel_calls_for_interleave=1,
        interleave_deterministic=True)
    generator = zdg.ZarrDataSetGenerator(
        root=zarr_root,
        batch_size=1,
        zarr_config=zarr_config,
        read_config=read_config)
    ds, stats = generator.as_dataset()
    height, width, num_in_channels, _, _ = stats

    ds = ds.take(1)
    l = list(ds.as_numpy_iterator())
    batched_examples = l[0]
    x = batched_examples[0]
    # The returned array should NOT be equal to np.ones(...).
    assert not np.array_equal(x, np.ones((1, height, width, num_in_channels)))


if __name__ == '__main__':
  absltest.main()
