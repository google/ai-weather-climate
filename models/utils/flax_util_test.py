import functools
import tempfile

from absl.testing import absltest
from ai_weather_climate.models.utils import flax_util
import flax
import tensorflow.data as tfd


class Dense(flax.linen.Module):

  @flax.linen.compact
  def __call__(self, x, train):
    del train
    return flax.linen.Dense(1)(x[0])


class FlaxUtilTest(absltest.TestCase):

  def test_checkpoints(self):
    train_ds = tfd.Dataset.from_tensor_slices([[[1.], [3]]])
    test_ds = tfd.Dataset.from_tensor_slices([[[2.], [5]]])

    model = Dense()

    # Bind args of train() that are going to be common between training
    # with checkpoints and without.
    train_f = functools.partial(
        flax_util.train,
        model,
        loss_fn=lambda x, y: flax_util.mse(x, y[0]),
        metrics_fn=lambda unused_1, unused_2: {'mse': 0.1},
        learning_rate=0.05,
        train_ds=train_ds,
        test_ds=test_ds)

    # Train for 2 steps checkpointing in between.
    with tempfile.TemporaryDirectory(prefix='model_') as model_dir:
      train_f(model_dir=model_dir, num_train_steps=1)
      train_state_actual = train_f(model_dir=model_dir, num_train_steps=2)

    # Train for 2 steps with no checkpoints.
    with tempfile.TemporaryDirectory(prefix='model_') as model_dir:
      train_state_expected = train_f(model_dir=model_dir, num_train_steps=2)

    # Replace numpy arrays with lists so we can use assertEqual on the
    # container.
    train_state_actual = train_state_actual.replace(
        step_rng=list(train_state_actual.step_rng))
    train_state_expected = train_state_expected.replace(
        step_rng=list(train_state_expected.step_rng))

    self.assertEqual(train_state_actual, train_state_expected)


if __name__ == '__main__':
  absltest.main()
