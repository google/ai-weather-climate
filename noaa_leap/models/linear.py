"""Linear model in flax for post-processing GFS."""
import datetime
import functools
import os.path
from typing import Callable, Iterable, Dict, Optional, Tuple

from absl import app
from absl import flags
from ai_weather_climate.noaa_leap.models import dataset_util
import flax
from flax.training import checkpoints
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import xarray

FLAGS = flags.FLAGS

flags.DEFINE_string('input_store',
                    '/bigstore/prj-leap-experimental/toy_2y_20210505-5.zarr',
                    'Path to gfs dataset.')
flags.DEFINE_string('checkpoint_dir', '/tmp/linear',
                    'Path to directory to store checkpoints.')

# Flax/jax sometimes uses generic types (Dict, Callable, etc.) without
# subscripting; so we disable the warning for that.
# pylint: disable=g-bare-generic


def identity_init(unused_key: jnp.ndarray,
                  shape: Iterable[int],
                  _=None) -> jnp.ndarray:
  shape = tuple(shape)
  assert len(shape) == 4, f'Expected shape to be of length 4, got {shape}'
  assert shape[0] == 1 and shape[1] == 1, (
      f'Expected dims 0 and 1 to have length 1, got {shape}')
  assert shape[2] == shape[3], (
      f'Expected dims 2 and 3 to have same length, got {shape}')
  return jnp.reshape(
      jnp.identity(shape[2], dtype=jnp.float32), (1, 1) + shape[2:])


class Linear(flax.linen.Module):
  """Linear model, i.e., 1x1 conv with no activation."""
  num_features: int

  def setup(self):
    self.conv_layer = flax.linen.Conv(
        features=self.num_features,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='VALID',
        kernel_init=identity_init)

  def __call__(self, features_by_name: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    return jax.vmap(
        self.conv_layer, in_axes=1, out_axes=1)(
            features_by_name['last_error'])


@flax.struct.dataclass
class TrainState:
  step: int
  optimizer: flax.optim.Optimizer
  loss: Optional[float]
  metrics: Optional[Dict[str, jnp.ndarray]]
  apply_fn: Callable = flax.struct.field(pytree_node=False)


def mse(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
  err = x - y
  return jnp.nanmean(err * err)


def compute_metrics(prediction: jnp.ndarray,
                    actual: jnp.ndarray) -> Dict[str, jnp.ndarray]:
  return {'mse': mse(prediction, actual), 'mae': jnp.abs(prediction - actual)}


def eval_step(state: TrainState,
              features_by_name: Dict[str, jnp.ndarray]) -> jnp.ndarray:
  predictions = state.apply_fn(state.optimizer.target, features_by_name)
  metrics = compute_metrics(predictions, features_by_name['error'])
  return jax.lax.pmean(metrics, axis_name='batch')


def eval_loop(state: TrainState, ds: tf.data.Dataset, ckpt_dir: str):
  """Computes and prints metrics on ds and saves params to a checkpoint."""
  eval_metrics = []
  p_eval_step = jax.pmap(eval_step, axis_name='batch')
  for batch in ds:
    batch = jax.tree_map(lambda x: np.reshape(x, (1,) + x.shape), batch)
    metrics = p_eval_step(state, batch)
    eval_metrics.append(metrics)

  # Take mean metrics (over the samples in a batch) from the first device - all
  # devices will have computed the same mean metrics.
  eval_metrics = jax.tree_map(lambda x: x[0], eval_metrics)

  # Transpose from list[dict[metric_name, batch_mean]] to
  # dict[metric_name, list[batch_mean]].
  eval_metrics = jax.tree_multimap(lambda *x: jnp.stack(x), *eval_metrics)

  # Compute mean over the dataset.
  metrics = jax.tree_map(lambda x: float(jnp.mean(x)), eval_metrics)
  print('Eval metrics:', metrics)
  checkpoints.save_checkpoint(
      ckpt_dir,
      flax.jax_utils.unreplicate(state).replace(metrics=metrics),
      state.step,
      prefix='cp_',
      keep=1,
      overwrite=False)


def loss_fn(apply_fn: Callable, features_by_name: Dict[str, jnp.ndarray],
            params: Dict) -> jnp.ndarray:
  out = apply_fn(params, features_by_name)
  return mse(out, features_by_name['error'])


@jax.jit
def train_step(state: TrainState, batch: Dict[str, jnp.ndarray]) -> TrainState:
  """Train for a single step."""

  def _loss_fn(params: Dict) -> jnp.ndarray:
    return loss_fn(state.apply_fn, batch, params)

  optimizer = state.optimizer
  grad_fn = jax.value_and_grad(_loss_fn, has_aux=False)
  loss, grad = grad_fn(optimizer.target)
  grad = jax.lax.pmean(grad, axis_name='batch')
  new_optimizer = optimizer.apply_gradient(grad)
  new_state = state.replace(
      optimizer=new_optimizer, loss=loss, step=state.step + 1)
  return new_state


def shard_to_devices(sample):
  device_count = jax.local_device_count()

  def _reshape(x):
    # Use _numpy() to convert between TF and NumPy without copying data.
    x = x._numpy()  # pylint: disable=protected-access
    return x.reshape((device_count, -1) + x.shape[1:])

  return jax.tree_map(_reshape, sample)


def train(model: flax.linen.Module, train_ds: tf.data.Dataset,
          test_ds: tf.data.Dataset, ckpt_dir: str,
          num_train_steps: int) -> TrainState:
  """Trains model on train_ds and evaluates it on test_ds."""
  sample = shard_to_devices(next(iter(train_ds)))

  params = model.init(jax.random.PRNGKey(1), sample)
  print('Model param shapes:\n',
        jax.tree_map(lambda x: x.shape, params)['params'])

  optimizer_def = flax.optim.Adam(learning_rate=0.05)
  optimizer = optimizer_def.create(params)

  state = TrainState(
      step=0,
      optimizer=optimizer,
      loss=None,
      metrics=None,
      apply_fn=model.apply)
  state = flax.jax_utils.replicate(state)

  p_train_step = jax.pmap(train_step, axis_name='batch')

  while state.step < num_train_steps:
    for batch in train_ds:
      batch = shard_to_devices(batch)
      if state.step % 10 == 0:
        eval_loop(state, test_ds, ckpt_dir)
      state = p_train_step(state, batch)
      print(f'Step: {state.step}, loss: {state.loss}')
  eval_loop(state, test_ds, ckpt_dir)
  return flax.jax_utils.unreplicate(state)


def clean_up_dataset(ds: xarray.Dataset) -> xarray.Dataset:
  ds = ds.drop_vars(['Cloud mixing ratio', 'Relative humidity'])
  for d in ds.dims:
    if d.startswith('z_'):
      if 1500 in ds[d]:
        ds = ds.drop_sel({d: 1500})
      if 4000 in ds[d]:
        ds = ds.drop_sel({d: 4000})
  return ds


def split_dataset(ds: xarray.Dataset,
                  timestamp: int) -> Tuple[xarray.Dataset, xarray.Dataset]:
  return ds.sel(t=slice(None, timestamp - 1)), ds.sel(t=slice(timestamp, None))


class Predictor:
  """A Callable that can be passed to pp_eval and that returns predictions."""

  def __init__(self, apply_fn, fc: int, ckpt_dir: str, max_nan_ratio: float):
    self.apply_fn = apply_fn
    self.fc = fc
    self.ckpt_dir = ckpt_dir
    self.max_nan_ratio = max_nan_ratio
    self.params = None

  def get_params(self):
    if self.params is None:
      state = flax.training.checkpoints.restore_checkpoint(
          self.ckpt_dir, target=None, prefix='cp_', parallel=True)
      self.params = state['optimizer']['target']
    return self.params

  def __call__(self, input_store: str,
               t: xarray.DataArray) -> Optional[xarray.Dataset]:
    ds = clean_up_dataset(xarray.open_zarr(input_store, consolidated=True))
    sample = dataset_util.get_sample(ds, t, self.fc)
    if not sample:
      return None
    sample_np = jax.tree_map(
        lambda x: jnp.expand_dims(x, axis=0),
        dataset_util.convert_to_nan_free_numpy(sample, self.max_nan_ratio))
    predicted_error = dataset_util.to_datasets(
        self.apply_fn(self.get_params(), sample_np)[0], [sample['error']])[0]
    return predicted_error + sample['forecast']


def main(unused_argv):
  # pylint: disable=g-long-lambda
  ds = clean_up_dataset(xarray.open_zarr(FLAGS.input_store, consolidated=True))
  train_ds, test_ds = split_dataset(ds, 1573063200)

  sample_gen = functools.partial(
      dataset_util.sample_generator, fc=6, max_nan_ratio=0.1, shuffle_seed=1)
  tf_train_ds = dataset_util.tf_dataset_from_iterable_f(
      lambda: sample_gen(train_ds)).batch(
          2, drop_remainder=True)
  tf_test_ds = dataset_util.tf_dataset_from_iterable_f(
      lambda: sample_gen(test_ds)).batch(
          2, drop_remainder=True).take(10)

  model = Linear(num_features=tf_train_ds.element_spec['error'].shape[-1])
  ckpt_dir = os.path.join(FLAGS.checkpoint_dir,
                          datetime.datetime.now().strftime('%Y%m%d_%H%M'))
  print(f'Checkpoint directory: {ckpt_dir}')
  train(model, tf_train_ds, tf_test_ds, ckpt_dir, num_train_steps=50)


if __name__ == '__main__':
  app.run(main)
