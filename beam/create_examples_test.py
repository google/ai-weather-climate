from absl import flags
from absl.testing import absltest
from ai_weather_climate.beam import create_examples
from ai_weather_climate.util import test_util
import numpy as np
import tensorflow as tf
import xarray
from xarray_beam._src import test_util as beam_test_util

FLAGS = flags.FLAGS


class CreateExamplesTest(absltest.TestCase):

  def setUp(self):
    super(CreateExamplesTest, self).setUp()
    self.tmp_dir = absltest.get_default_test_tmpdir()

  def test_main(self):
    # Create input.
    input_zarr = test_util.create_test_zarr_cubesphere_dataset(self.tmp_dir)
    in_ds = xarray.open_zarr(input_zarr, consolidated=True)

    # Create stats.
    in_ds_group = in_ds.groupby('time.dayofyear')
    stats_mean = in_ds_group.mean(dim=['time']).expand_dims(stat=['mean'])
    stats_std = in_ds_group.std(dim=['time']).expand_dims(stat=['std'])
    stats_ds = xarray.concat((stats_mean, stats_std), 'stat')

    # Statics.
    static_zarr = test_util.create_test_static_cubesphere_dataset(self.tmp_dir)
    static_ds = xarray.open_zarr(static_zarr, consolidated=True)

    # Create a sample.
    make_examples = create_examples.MakeExamples(in_ds, stats_ds, static_ds)
    sample = make_examples.make_sample(in_ds.time[0].data)
    self.assertEqual((6, 48, 48, 9), sample.shape)

    # Create an example.
    t = in_ds.time.data[0]
    example_str = make_examples.make_example(t)
    example = tf.train.Example()
    example.ParseFromString(example_str)

    # Parse into a dictionary.
    result = {}
    for key, feature in example.features.feature.items():
      # The values are the Feature objects which contain a `kind` which
      # contains: one of three fields: bytes_list, float_list, int64_list
      kind = feature.WhichOneof('kind')
      result[key] = np.array(getattr(feature, kind).value)
    actual_sample = result['input'].reshape(sample.shape)
    np.testing.assert_almost_equal(sample, actual_sample)
    key = np.datetime64(int(result['key'][0]), 's')
    self.assertEqual(t, key)

    # Run the pipeline
    examples = make_examples.expand(beam_test_util.EagerPipeline())
    self.assertEqual(len(in_ds.time), len(examples))


if __name__ == '__main__':
  absltest.main()
