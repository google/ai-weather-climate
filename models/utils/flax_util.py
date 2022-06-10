"""Train and eval loops for flax and a some flax helpers."""
import json
import logging
import os
import sys
import time
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import flax
from flax.metrics import tensorboard
from flax.training import checkpoints
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import tensorflow as tf
from tensorflow.io import gfile

# Flax/jax sometimes uses generic types (Dict, Callable, etc.) without
# subscripting; so we disable the warning for that.
# pylint: disable=g-bare-generic


def log(msg: str, *args) -> None:
  try:
    is_tty = os.isatty(sys.stdin.fileno())
  except AttributeError:
    # Colab env assigns an object to sys.stdin that doesn't have fileno().
    is_tty = True
  if is_tty:
    print(msg % args)
  else:
    logging.info(msg, *args)


class _Timer:
  """Simple timer."""

  def __init__(self):
    self._start_time = time.time()
    self._elapsed_time = 0

  def get_elapsed_time(self) -> float:
    return self._elapsed_time + (
        time.time() - self._start_time if self._start_time is not None else 0)

  def stop(self) -> None:
    if self._start_time is not None:
      self._elapsed_time += time.time() - self._start_time
      self._start_time = None

  def start(self) -> None:
    self._start_time = time.time()

  def reset(self) -> None:
    if self._start_time is not None:
      self._start_time = time.time()
    self._elapsed_time = 0


@flax.struct.dataclass
class TrainState:
  step: int
  opt_state: Any
  params: Any
  step_rng: Any
  loss: Optional[float]


@flax.struct.dataclass
class Checkpoint:
  metrics: Dict[str, float]
  state: TrainState


def save_metadata(metadata: ml_collections.ConfigDict, model_dir: str) -> None:
  gfile.makedirs(model_dir)
  with gfile.GFile(os.path.join(model_dir, 'metadata.json'), 'w') as f:
    f.write(metadata.to_json())


def load_metadata(model_dir: str) -> ml_collections.ConfigDict:
  with gfile.GFile(os.path.join(model_dir, 'metadata.json'), 'r') as f:
    return ml_collections.ConfigDict(json.load(f))


def save_checkpoint(state: TrainState, metrics: Dict[str, float],
                    model_dir: str) -> None:
  checkpoints.save_checkpoint(
      model_dir,
      Checkpoint(metrics=metrics, state=state),
      state.step,
      prefix='cp_',
      keep=1,
      overwrite=False)


def init_train_state(
    model_dir: str, model_init_fn: Callable, eval_loop_fn: Callable,
    learning_rate: float, sample_ds: tf.data.Dataset
) -> Tuple[bool, TrainState, optax.GradientTransformation, Dict[str, float]]:
  """Initializes training state from a checkpoint or with defaults."""
  sample = next(sample_ds.as_numpy_iterator())
  log('Sample shapes:\n%s\n', jax.tree_map(lambda x: x.shape, sample))

  params = model_init_fn(sample)
  optimizer = optax.adam(learning_rate=learning_rate)
  opt_state = optimizer.init(params)
  state = TrainState(step=0, opt_state=opt_state, params=params,
                     step_rng=jax.random.PRNGKey(1), loss=None)
  rep_state = flax.jax_utils.replicate(state)
  metrics = eval_loop_fn(rep_state, sample_ds.take(1))
  checkpoint_template = Checkpoint(metrics=metrics, state=state)
  checkpoint = flax.training.checkpoints.restore_checkpoint(
      model_dir, target=checkpoint_template, prefix='cp_', parallel=True)
  if checkpoint.state.loss is None:
    return False, state, optimizer, {}
  params = checkpoint.state.params
  log('Model param shapes:\n%s\n', jax.tree_map(lambda x: x.shape, params))

  return True, checkpoint.state, optimizer, checkpoint.metrics


def default_model_init(model):
  return lambda sample: model.init(jax.random.PRNGKey(1), sample, train=True)


def restore_checkpoint(model_dir: str, model: flax.linen.Module,
                       metrics_fn: Callable,
                       sample_ds: tf.data.Dataset) -> TrainState:
  """Restores TrainState from last checkpoint in model_dir."""
  return restore_checkpoint_custom(
      model_dir,
      default_model_init(model),
      lambda state, ds: eval_loop(model.apply, metrics_fn, state, ds),
      sample_ds)


def restore_checkpoint_custom(model_dir: str, model_init_fn: Callable,
                              eval_loop_fn: Callable,
                              sample_ds: tf.data.Dataset) -> TrainState:
  """Restores TrainState from last checkpoint in model_dir."""
  from_checkpoint, train_state, _, _ = (
      init_train_state(
          model_dir,
          model_init_fn,
          eval_loop_fn,
          learning_rate=0,
          sample_ds=sample_ds))
  if not from_checkpoint:
    raise ValueError(f'Checkpoint not found in {model_dir}')
  return train_state


def get_batch_size(batch) -> int:
  if isinstance(batch, dict):
    return sum(get_batch_size(v) for v in batch.values())
  elif isinstance(batch, tuple):
    return sum(get_batch_size(v) for v in batch)
  return np.prod(batch.shape)


def shard_to_devices(sample):
  device_count = jax.local_device_count()

  def _reshape(x):
    # Use _numpy() to convert between TF and NumPy without copying data.
    x = x._numpy()  # pylint: disable=protected-access
    return x.reshape((device_count, -1) + x.shape[1:])

  return jax.tree_map(_reshape, sample)


def mse(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
  err = x - y
  return jnp.nanmean(err * err)


def write_dict_to_tensorboard(prefix: str, d: Dict[str, float], step: int,
                              w: tensorboard.SummaryWriter) -> None:
  for k, v in d.items():
    w.scalar(prefix + k, v, step)


def eval_loop(apply_fn: Callable, metrics_fn: Callable, state: TrainState,
              ds: tf.data.Dataset) -> Dict[str, float]:
  """Computes metrics on ds."""

  @jax.jit
  def eval_step(state: TrainState,
                features_by_name: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    predictions = apply_fn(state.params, features_by_name, train=False)
    metrics = metrics_fn(predictions, features_by_name)
    return jax.lax.pmean(metrics, axis_name='batch')

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
  log('Eval metrics: %s', metrics)
  return metrics


def create_train_step(
    apply_fn: Callable, loss_fn: Callable,
    optimizer: optax.GradientTransformation
) -> Callable[[TrainState, Dict[str, jnp.ndarray]], TrainState]:
  """Returns a jitted train_step function."""

  def f(state: TrainState, batch: Dict[str, jnp.ndarray]) -> TrainState:

    def _loss_fn(params: Dict) -> jnp.ndarray:
      predictions = apply_fn(params, batch, train=True)
      return loss_fn(predictions, batch)

    grad_fn = jax.value_and_grad(_loss_fn, has_aux=False)
    loss, grad = grad_fn(state.params)
    grad = jax.lax.pmean(grad, axis_name='batch')
    updates, opt_state = optimizer.update(grad, state.opt_state, state.params)
    params = optax.apply_updates(state.params, updates)
    new_state = state.replace(
        params=params, opt_state=opt_state, loss=loss, step=state.step + 1)
    return new_state

  return jax.jit(f)


def train(model: flax.linen.Module, loss_fn: Callable, metrics_fn: Callable,
          learning_rate: float, train_ds: tf.data.Dataset,
          test_ds: tf.data.Dataset, model_dir: str,
          num_train_steps: int, update_freq: int = 10) -> TrainState:

  def get_p_train_step(optimizer):
    return jax.pmap(
        create_train_step(model.apply, loss_fn, optimizer), axis_name='batch')

  return train_custom(
      model_init_fn=default_model_init(model),
      get_p_train_step=get_p_train_step,
      eval_loop_fn=lambda state, ds: eval_loop(
          model.apply, metrics_fn, state, ds),
      learning_rate=learning_rate,
      train_ds=train_ds,
      test_ds=test_ds,
      model_dir=model_dir,
      num_train_steps=num_train_steps,
      update_freq=update_freq)


def train_custom(model_init_fn: Callable,
                 get_p_train_step: Callable,
                 eval_loop_fn: Callable,
                 learning_rate: float,
                 train_ds: tf.data.Dataset,
                 test_ds: tf.data.Dataset,
                 model_dir: str,
                 num_train_steps: int,
                 update_freq: int = 10) -> TrainState:
  """Trains model on train_ds and evaluates it on test_ds."""
  last_checkpoint_step = None

  log('Checkpoint directory: %s', model_dir)
  from_checkpoint, state, optimizer, metrics = init_train_state(
      model_dir, model_init_fn, eval_loop_fn, learning_rate, train_ds)
  if from_checkpoint:
    log('Loaded params from checkpoint dir: %s', model_dir)
    log('Checkpoint metrics at step %s: %s', state.step, metrics)
    last_checkpoint_step = state.step

  log('Model param shapes:\n%s\n',
      jax.tree_map(lambda x: x.shape, state.params)['params'])
  log(
      'Num params: %s',
      jax.tree_util.tree_reduce(lambda r, x: r + jnp.prod(jnp.array(x.shape)),
                                state.params, 0))

  rep_state = flax.jax_utils.replicate(state)
  summary_writer = tensorboard.SummaryWriter(model_dir)

  p_train_step = get_p_train_step(optimizer)

  train_timer = _Timer()
  while state.step < num_train_steps:
    log('Starting epoch.')
    for batch in train_ds:
      batch_size = get_batch_size(batch)
      batch = shard_to_devices(batch)
      if last_checkpoint_step != state.step and state.step % update_freq == 0:
        train_timer.stop()
        eval_timer = _Timer()
        metrics = eval_loop_fn(rep_state, test_ds)
        summary_writer.scalar('perf/eval duration (s)',
                              eval_timer.get_elapsed_time(), state.step)
        write_dict_to_tensorboard('eval/', metrics, state.step, summary_writer)
        save_checkpoint(state, metrics, model_dir)
        last_checkpoint_step = state.step
        train_timer.start()
      rep_state = p_train_step(rep_state, batch)
      _, step_rng = jax.random.split(rep_state.step_rng[0], 2)
      rep_state = rep_state.replace(
          step_rng=jnp.tile(step_rng[None, :],
                            (rep_state.step_rng.shape[0], 1)))
      state = flax.jax_utils.unreplicate(rep_state)
      summary_writer.scalar('loss', state.loss, state.step)
      summary_writer.scalar(
          'perf/training data rate (Mfloat/s)',
          batch_size / 1024**2 / train_timer.get_elapsed_time(), state.step)
      summary_writer.scalar('perf/training step rate (1/sec)',
                            1 / train_timer.get_elapsed_time(), state.step)
      train_timer.reset()
      log('Step: %s, loss: %s', state.step, state.loss)
      if jnp.isnan(state.loss):
        raise ValueError(f'Loss is nan at step {state.step}!')
      if state.step >= num_train_steps:
        break
  metrics = eval_loop_fn(rep_state, test_ds)
  if last_checkpoint_step != state.step:
    write_dict_to_tensorboard('eval/', metrics, state.step, summary_writer)
    save_checkpoint(state, metrics, model_dir)

  summary_writer.close()
  return state


def apply_model(apply_fn: Callable[[jnp.ndarray], jnp.ndarray],
                sample_ds: tf.data.Dataset) -> Iterable[jnp.ndarray]:
  for batch in sample_ds.as_numpy_iterator():
    yield apply_fn(batch)
