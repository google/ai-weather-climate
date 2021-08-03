import os.path

from absl import flags
from absl.testing import absltest
from ai_weather_climate.noaa_leap.models import dataset_test_util
from ai_weather_climate.noaa_leap.models import linear
import numpy as np
import tensorflow as tf

FLAGS = flags.FLAGS


class LinearTest(absltest.TestCase):

  def test_train(self):
    rs = np.random.RandomState(0)
    sample = rs.uniform(size=(2, 4, 3, 1))  # batch, i, j, channels
    # Dataset for learning the identity function.
    train_ds = tf.data.Dataset.from_tensor_slices({
        'last_error': [sample],
        'error': [sample]
    })
    test_ds = train_ds
    model = linear.Linear(num_features=1)
    state = linear.train(
        model, train_ds, test_ds, ckpt_dir=FLAGS.test_tmpdir, num_train_steps=1)
    pred = model.apply(state.optimizer.target,
                       next(test_ds.as_numpy_iterator()))
    # The linear model inits itself as the identity function, so we can check
    # that the predictions are exactly correct (and confirm that training didn't
    # do anything silly).
    np.testing.assert_allclose(pred, sample)

  def test_predictor(self):
    ds, tf_ds = dataset_test_util.create_fake_gfs()

    ds_path = os.path.join(FLAGS.test_tmpdir, 'ds')
    ds.to_zarr(ds_path, consolidated=True)

    model = linear.Linear(num_features=1)

    ckpt_dir = os.path.join(FLAGS.test_tmpdir, 'checkpoints')
    state = linear.train(
        model, tf_ds, tf_ds, ckpt_dir=ckpt_dir, num_train_steps=1)

    # Compute the expected prediction by applying the model directly to the
    # first sample in the dataset using the params returned by train().
    sample = next(tf_ds.as_numpy_iterator())
    expected_error = model.apply(state.optimizer.target, sample)
    expected_prediction = expected_error + sample['forecast']

    # The predictor reads the model params from the checkpoint (saved by train)
    # and the input features from the zarr file.
    predictor = linear.Predictor(
        model.apply, fc=1, ckpt_dir=ckpt_dir, max_nan_ratio=0.1)
    # The time index for the sample we read above is 2. See
    # dataset_util.get_sample() for details.
    actual_prediction = predictor(ds_path, ds.t.isel(t=2))

    np.testing.assert_allclose(
        np.squeeze(expected_prediction, axis=(0, -1)),
        actual_prediction['Temperature'])


if __name__ == '__main__':
  absltest.main()
