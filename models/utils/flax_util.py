"""Train and eval loops for flax and a some flax helpers."""
import dataclasses
import functools
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
  state = TrainState(
      step=0,
      opt_state=opt_state,
      params=params,
      step_rng=jax.random.PRNGKey(1),
      loss=None)
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
  rng = jax.random.PRNGKey(1)
  return lambda sample: jax.jit(model.init)(rng, sample, train=True)


def restore_checkpoint(model_dir: str, model: flax.linen.Module,
                       metrics_fn: Callable,
                       sample_ds: tf.data.Dataset) -> TrainState:
  """Restores TrainState from last checkpoint in model_dir."""
  return restore_checkpoint_custom(
      model_dir, default_model_init(model),
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


def get_num_floats_in_batch(batch) -> int:
  if isinstance(batch, dict):
    return sum(get_num_floats_in_batch(v) for v in batch.values())
  elif isinstance(batch, tuple):
    return sum(get_num_floats_in_batch(v) for v in batch)
  return np.prod(batch.shape)


def get_num_samples_in_batch(batch) -> int:
  if isinstance(batch, dict):
    return get_num_samples_in_batch(next(iter(batch.values())))
  elif isinstance(batch, tuple):
    return get_num_samples_in_batch(next(iter(batch)))
  return batch.shape[0]


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
  eval_metrics = jax.tree_map(lambda *x: jnp.stack(x), *eval_metrics)

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


def train(model: flax.linen.Module,
          loss_fn: Callable,
          metrics_fn: Callable,
          learning_rate: float,
          train_ds: tf.data.Dataset,
          test_ds: tf.data.Dataset,
          model_dir: str,
          num_train_steps: int,
          metrics_update_secs: int = 60) -> TrainState:
  """Trains model; uses the default initializer and eval loop in the process."""

  def get_p_train_step(optimizer):
    return jax.pmap(
        create_train_step(model.apply, loss_fn, optimizer), axis_name='batch')

  # pylint: disable=g-long-lambda
  return train_custom(
      model_init_fn=default_model_init(model),
      get_p_train_step=get_p_train_step,
      eval_loop_fn=lambda state, ds: eval_loop(model.apply, metrics_fn, state,
                                               ds),
      learning_rate=learning_rate,
      train_ds=train_ds,
      test_ds=test_ds,
      model_dir=model_dir,
      num_train_steps=num_train_steps,
      metrics_update_secs=metrics_update_secs)


_AVG_TRAIN_LOSS = 'average train loss'


@dataclasses.dataclass
class PerfCounters:
  """Counters tracking training performance."""
  batch_count: int = 0
  sample_count: int = 0
  sample_size_bytes: int = 0
  train_loss_total: float = 0

  def to_dict(self, duration_sec: float):
    if self.batch_count == 0 or duration_sec == 0:
      return {}
    return {
        'data rate (Mfloat/s)':
            (self.sample_size_bytes / 1024**2 / duration_sec),
        # 'samples per batch' is the number of samples in a batch
        # averaged over some training steps (e.g., the epoch or between
        # evals). Note that normally every batch has the same number of
        # samples, so the average will not vary over time.
        'samples per batch': (self.sample_count / self.batch_count),
        'sample rate (1/sec)': (self.sample_count / duration_sec),
        _AVG_TRAIN_LOSS: (self.train_loss_total / self.batch_count),
    }

  def add(self, other: 'PerfCounters'):
    self.batch_count += other.batch_count
    self.sample_count += other.sample_count
    self.sample_size_bytes += other.sample_size_bytes
    self.train_loss_total += other.train_loss_total


def _update_metrics(state: TrainState, model_dir: str,
                    summary_writer: tensorboard.SummaryWriter,
                    eval_loop_fn: Callable, update_metrics_timer: _Timer,
                    perf_counters: PerfCounters, should_save_checkpoint: bool):
  """Runs eval, updates tensorboard, and saves state to a checkpoint."""
  eval_timer = _Timer()
  eval_metrics = eval_loop_fn()
  eval_timer.stop()
  secs_since_last = update_metrics_timer.get_elapsed_time()
  update_metrics_timer.reset()

  eval_duration_ratio = eval_timer.get_elapsed_time() / secs_since_last

  summary_writer.scalar('eval/eval time %', eval_duration_ratio * 100,
                        state.step)
  write_dict_to_tensorboard('eval/', eval_metrics, state.step, summary_writer)
  if should_save_checkpoint:
    save_checkpoint(state, eval_metrics, model_dir)

  perf_metrics_dict = perf_counters.to_dict(secs_since_last)
  write_dict_to_tensorboard('train/', perf_metrics_dict, state.step,
                            summary_writer)


def train_custom(model_init_fn: Callable,
                 get_p_train_step: Callable,
                 eval_loop_fn: Callable,
                 learning_rate: float,
                 train_ds: tf.data.Dataset,
                 test_ds: tf.data.Dataset,
                 model_dir: str,
                 num_train_steps: int,
                 metrics_update_secs: float = 60) -> TrainState:
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

  update_metrics_timer = _Timer()
  train_perf_counters = PerfCounters()
  while state.step < num_train_steps:
    log('Starting epoch.')
    epoch_timer = _Timer()
    epoch_perf_counters = PerfCounters()
    for batch in train_ds:
      device_batch = shard_to_devices(batch)
      if update_metrics_timer.get_elapsed_time() > metrics_update_secs:
        _update_metrics(
            state,
            model_dir,
            summary_writer,
            functools.partial(eval_loop_fn, rep_state, test_ds),
            update_metrics_timer,
            train_perf_counters,
            should_save_checkpoint=last_checkpoint_step != state.step)
        last_checkpoint_step = state.step
        train_perf_counters = PerfCounters()

      rep_state = p_train_step(rep_state, device_batch)
      _, step_rng = jax.random.split(rep_state.step_rng[0], 2)
      rep_state = rep_state.replace(
          step_rng=jnp.tile(step_rng[None, :], (rep_state.step_rng.shape[0],
                                                1)))
      state = flax.jax_utils.unreplicate(rep_state)
      if jnp.isnan(state.loss):
        raise ValueError(f'Loss is nan at step {state.step}!')
      batch_counters = PerfCounters(
          batch_count=1,
          sample_size_bytes=get_num_floats_in_batch(batch),
          sample_count=get_num_samples_in_batch(batch),
          train_loss_total=state.loss)
      train_perf_counters.add(batch_counters)
      epoch_perf_counters.add(batch_counters)
      if state.step >= num_train_steps:
        break
    epoch_perf_metrics_dict = epoch_perf_counters.to_dict(
        update_metrics_timer.get_elapsed_time())
    write_dict_to_tensorboard('epoch/', epoch_perf_metrics_dict, state.step,
                              summary_writer)
    epoch_timer.reset()
    log('Average batch loss in epoch: %.2f after %d steps',
        epoch_perf_metrics_dict[_AVG_TRAIN_LOSS], state.step)
  _update_metrics(
      state,
      model_dir,
      summary_writer,
      functools.partial(eval_loop_fn, rep_state, test_ds),
      update_metrics_timer,
      train_perf_counters,
      should_save_checkpoint=last_checkpoint_step != state.step)
  summary_writer.close()
  return state


def apply_model(apply_fn: Callable[[jnp.ndarray], jnp.ndarray],
                sample_ds: tf.data.Dataset) -> Iterable[jnp.ndarray]:
  for batch in sample_ds.as_numpy_iterator():
    yield apply_fn(batch)
