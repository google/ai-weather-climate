"""Techniques for determining the saliency of input features."""

import logging
from typing import List, Optional, Tuple, Union

from ai_weather_climate.interpretability import utils
import numpy as np
import tensorflow as tf


def get_gradients_against_prediction(
    input_img: Union[tf.Tensor, np.ndarray],
    model: tf.keras.Model,
    mask: Optional[tf.Tensor] = None,
    has_batch_dim: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
  """Computes the gradients of the given input against prediction.

  Args:
    input_img: The input tensor. The dimensions of the tensor should match what
      the model expects except for the batch dimension. For eg., if model
      expects a (b, h, w, c) tensor, the input should be (h, w, c).
    model: The trained Keras model.
    mask: The mask to apply to the prediction before obtaining gradients.
    has_batch_dim: Whether the model input and output has a batch dimension.

  Returns:
    Returns gradients with respect to the input of same shape as input.
  """
  input_img = tf.convert_to_tensor(input_img)

  if has_batch_dim:
    # Add the batch dimension.
    input_img = tf.expand_dims(input_img, axis=0)

  with tf.GradientTape() as gtape:
    gtape.watch(input_img)
    prediction = model(input_img)
    if has_batch_dim:
      # Remove the batch dimension.
      prediction = prediction[0, ...]
    if mask is not None:
      assert mask.shape == prediction.shape, (f'{mask.shape} is not equal to '
                                              f'{prediction.shape}')
      prediction = tf.multiply(mask, prediction)
  grads = gtape.gradient(prediction, input_img)

  if has_batch_dim:
    # Remove the batch dimension.
    grads = np.squeeze(grads, axis=0)

  return grads, prediction


def get_gradients_against_loss(
    input_img: tf.Tensor,
    model: tf.keras.Model,
    label: tf.Tensor,
    mask: Optional[tf.Tensor] = None,
    has_batch_dim: bool = True,
    is_classification: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
  """Computes the gradients of the given input against loss.

  Args:
    input_img: The input tensor. The dimensions of the tensor should match what
      the model expects except for the batch dimension. For eg., if model
      expects a (b, h, w, c) tensor, the input should be (h, w, c).
    model: The trained Keras model.
    label: The actual output.
    mask: The mask to apply to the prediction before obtaining gradients.
    has_batch_dim: Whether the model input and output has a batch dimension.
    is_classification: Whether the model being interpreted is a classification
      model.

  Returns:
    Returns gradients of loss with respect to the input of same shape as input.
  """
  input_img = tf.convert_to_tensor(input_img)

  if has_batch_dim:
    # Add the batch dimension.
    input_img = tf.expand_dims(input_img, axis=0)

  with tf.GradientTape() as gtape:
    gtape.watch(input_img)
    prediction = model(input_img)
    if has_batch_dim:
      # Remove the batch dimension.
      prediction = prediction[0, ...]
    assert prediction.shape == label.shape
    # TODO(shreyaa): Add support for loss functions other than cross entropy.
    if mask is not None:
      assert mask.shape == prediction.shape, (f'{mask.shape} is not equal to '
                                              f'{prediction.shape}')
      prediction = tf.multiply(mask, prediction)
      label = tf.multiply(mask, label)
    if is_classification:
      loss = tf.keras.losses.binary_crossentropy(label, prediction)
    else:
      loss = tf.keras.losses.mean_squared_error(label, prediction)

  grads = gtape.gradient(loss, input_img)

  if has_batch_dim:
    # Remove the batch dimension.
    grads = np.squeeze(grads, axis=0)

  return grads, loss


def sanity_check_integrated_gradients(integrated_gradients: np.ndarray,
                                      predictions: List[np.ndarray]):
  """Sanity checks an integrated gradients computation.

  Ideally, the sum of the integrated gradients is equal to the difference in the
  predictions at the input and baseline. Any discrepancy in these two values is
  due to the errors in approximating the integral.

  Args:
    integrated_gradients: Integrated gradients for an input and predictions.
    predictions: The predicted probability distribution across all classes for
      the various inputs considered in computing integrated gradients. It has
      shape <steps, shape of each predictions> where 'steps' is the number of
      integrated gradient steps.
  """
  want_integral = np.sum(predictions[-1] - predictions[0])
  got_integral = np.sum(integrated_gradients)
  if want_integral == 0.0:
    raise ValueError(
        'FAIL: The prediction at the input is equal to that at the '
        'baseline. Please use a different baseline. Some '
        'suggestions are: random input, mean of the training set.')

  diff = 100.0 * abs(want_integral - got_integral) / abs(want_integral)
  if diff > 5.0:
    raise ValueError('FAIL: Integral approximation error is too high: {} '
                     'percent (obtained integral: {}, expected integral: {}). '
                     'The acceptable limit is 5 percent. Please try increasing '
                     'the number of integrated gradient steps'.format(
                         diff, got_integral, want_integral))
  logging.info(
      'Integral approximation error is %s percent which is within the '
      'acceptable threshold of 5 percent', diff)


def get_integrated_gradients(
    input_img: Union[tf.Tensor, np.ndarray],
    model: tf.keras.Model,
    label: Optional[Union[tf.Tensor, np.ndarray]] = None,
    mask: Optional[tf.Tensor] = None,
    baseline: Optional[np.ndarray] = None,
    num_steps: int = 50,
    check_sanity: bool = True,
    gradient_base: str = 'prediction',
    is_classification: bool = True,
) -> Optional[tf.Tensor]:
  """Computes Integrated Gradients for a predicted label.

  Original paper https://arxiv.org/pdf/1703.01365.pdf. The rough idea is to take
  a straight line path from the baseline to the input and compute gradients at
  several points along the path. Integrated gradients are obtained by taking the
  cumulative (integral) of these gradients.

  Args:
    input_img: The input tensor. The dimensions of the tensor should match what
      the model expects except for the batch dimension. For example, if model
      expects a (b, h, w, c) tensor, the input should be (h, w, c).
    model: The trained Keras model.
    label: The actual output.
    mask: The mask to apply to the prediction before obtaining gradients.
    baseline: The baseline image to start with for interpolation. If None, an
      image of zeros is used.
    num_steps: Number of interpolation steps between the baseline and the input
      used in the computation of integrated gradients. These steps along
      determine the integral approximation error. By default, num_steps is set
      to 50.
    check_sanity: Whether to perform a sanity check of the integrated gradient
      values.
    gradient_base: Specifies what the gradient of inputs should be computed
      against.
    is_classification: Whether the model being interpreted is a classification
      model.

  Returns:
    Returns integrated gradients with respect to the input of same shape as
    input.
  """
  # If baseline is not provided, start with a black image having same size as
  # the input image.
  if baseline is None:
    baseline = np.zeros(input_img.shape)

  baseline = baseline.astype(np.float32)
  input_img = tf.cast(input_img, dtype=tf.float32)

  # 1. Do the interpolation for given number of steps.
  interpolated_image = np.linspace(baseline, input_img, num_steps + 1)

  # 2. Get the gradients.
  grads = []
  outputs = []
  for img in interpolated_image:
    if gradient_base == 'loss' and label is not None:
      grad, output = get_gradients_against_loss(
          img,
          model,
          label=label,
          mask=mask,
          is_classification=is_classification)
    else:
      grad, output = get_gradients_against_prediction(img, model, mask=mask)
    grads.append(grad)
    outputs.append(output)
  grads = tf.convert_to_tensor(grads, dtype=tf.float32)

  # 3. Approximate the integral using the trapezoidal rule.
  grads = (grads[:-1] + grads[1:]) / 2.0
  avg_grads = tf.reduce_mean(grads, axis=0)

  # 4. Calculate integrated gradients and return.
  final_ig = tf.multiply((input_img - baseline), avg_grads)
  if check_sanity:
    try:
      sanity_check_integrated_gradients(final_ig, outputs)
    except ValueError:
      return None
  return final_ig


def get_integrated_gradients_with_retry(
    input_img: Union[tf.Tensor, np.ndarray],
    model: tf.keras.Model,
    label: Optional[Union[tf.Tensor, np.ndarray]] = None,
    mask: Optional[tf.Tensor] = None,
    baseline: Optional[np.ndarray] = None,
    retry_times: int = 3,
    gradient_base: str = 'prediction',
    is_classification: bool = True,
    num_steps: int = 50,
  ) -> Optional[tf.Tensor]:
  """Returns sanity checked IG by doubling the number of steps in each retry."""
  for _ in range(retry_times):
    ig = get_integrated_gradients(
        input_img=input_img,
        model=model,
        label=label,
        mask=mask,
        baseline=baseline,
        num_steps=num_steps,
        gradient_base=gradient_base,
        is_classification=is_classification)
    if ig is None:
      num_steps = num_steps * 2
    else:
      return ig
  return None


def random_baseline_integrated_gradients(input_img: Union[tf.Tensor,
                                                          np.ndarray],
                                         model: tf.keras.Model,
                                         mask: Optional[tf.Tensor] = None,
                                         num_steps: int = 20,
                                         num_runs: int = 5) -> tf.Tensor:
  """Computes Integrated Gradients using a random baseline.

  Args:
    input_img: The input tensor. The dimensions of the tensor should match what
      the model expects except for the batch dimension. For eg., if model
      expects a (b, h, w, c) tensor, the input should be (h, w, c).
    model: The trained Keras model.
    mask: The mask to apply to the prediction before obtaining gradients.
    num_steps: Number of interpolation steps between the baseline and the input
      used in the computation of integrated gradients. These steps along
      determine the integral approximation error. By default, num_steps is set
      to 50.
    num_runs: Number of runs to do this over to pick different baselines.

  Returns:
    Averaged integrated gradients for `num_runs` baseline images.
  """
  # 1. Get the integrated gradients for all the baselines.
  integrated_grads = []
  for _ in range(num_runs):
    baseline = np.random.uniform(
        low=np.amin(input_img), high=np.amax(input_img), size=input_img.shape)
    igrads = get_integrated_gradients(
        input_img=input_img,
        model=model,
        mask=mask,
        baseline=baseline,
        num_steps=num_steps,
    )
    integrated_grads.append(igrads)

  # 2. Return the average integrated gradients for the image over all the runs.
  integrated_grads = tf.convert_to_tensor(integrated_grads, dtype=tf.float32)
  return tf.reduce_mean(integrated_grads, axis=0)


def get_blur_integrated_gradients(
    input_img: Union[tf.Tensor, np.ndarray],
    model: tf.keras.Model,
    mask: Optional[tf.Tensor] = None,
    grad_step: float = 0.01,
    max_sigma: float = 50,
    num_steps: int = 50,
    check_sanity: bool = True) -> Optional[np.ndarray]:
  """Computes Integrated Gradients for a predicted label by blurring the inputs.

  Original paper:
  https://openaccess.thecvf.com/content_CVPR_2020/papers/Xu_Attribution_in_Scale_and_Space_CVPR_2020_paper.pdf.
  One of the main advantages of this technique over the standard integrated
  gradients is that it does not require a baseline.

  Args:
    input_img: The input tensor. The dimensions of the tensor should match what
      the model expects except for the batch dimension. For example, if model
      expects a (b, h, w, c) tensor, the input should be (h, w, c).
    model: The trained Keras model.
    mask: The mask to apply to the prediction before obtaining gradients.
    grad_step: Gaussian gradient step size.
    max_sigma: The maximum value of sigma to be used for blurring.
    num_steps: Number of steps of blurring the input image. These steps along
      determine the integral approximation error. By default, num_steps is set
      to 50.
    check_sanity: Whether to perform a sanity check of the integrated gradient
      values.

  Returns:
    Returns integrated gradients with respect to the input of same shape as
    input.
  """
  sigmas = np.linspace(0, max_sigma, num_steps + 1)

  step_vector_diff = np.diff(sigmas)

  total_gradients = np.zeros_like(input_img)
  preds = []
  for i in range(num_steps):
    blurred_img = utils.gaussian_blur(input_img, sigmas[i])
    gaussian_gradient = (utils.gaussian_blur(
        input_img, sigmas[i] + grad_step) - blurred_img) / grad_step
    grad, pred = get_gradients_against_prediction(blurred_img, model, mask=mask)
    total_gradients += step_vector_diff[i] * np.multiply(
        gaussian_gradient, grad)
    preds.append(pred)

  if check_sanity:
    try:
      sanity_check_integrated_gradients(total_gradients, preds)
    except ValueError:
      return None
  return total_gradients * -1.0


def get_blur_integrated_gradients_with_retry(
    input_img: Union[tf.Tensor, np.ndarray],
    model: tf.keras.Model,
    mask: Optional[tf.Tensor] = None,
    grad_step: float = 0.01,
    max_sigma: float = 50,
    retry_times: int = 3) -> Optional[np.ndarray]:
  """Returns sanity checked Blur IG by doubling steps in every retry."""
  num_steps = 50
  for _ in range(retry_times):
    ig = get_blur_integrated_gradients(
        input_img=input_img,
        model=model,
        mask=mask,
        grad_step=grad_step,
        max_sigma=max_sigma,
        num_steps=num_steps)
    if ig is None:
      num_steps = num_steps * 2
    else:
      return ig
  return None
