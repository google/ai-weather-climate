from typing import Tuple

from ai_weather_climate.interpretability import saliency_maps
import numpy as np
import tensorflow.compat.v2 as tf


def _get_fake_model(input_shape: Tuple[int, ...], output_channels: int):
  img_input = tf.keras.layers.Input(shape=input_shape)
  net = img_input
  for i, depth in enumerate([2, 4, output_channels]):
    # The convolution layer downsamples the input.
    net = tf.keras.layers.Conv2D(
        filters=depth,
        kernel_size=(2, 2),
        strides=(1, 1),
        padding='same',
        use_bias=True,
        name=f'Conv2d_{i}')(
            net)
    net = tf.keras.layers.Activation('sigmoid')(net)

  return tf.keras.models.Model(img_input, net, name='model')


class SaliencyMapsTest(tf.test.TestCase):
  _IMAGE_1 = np.array([[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
                       [[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]]])
  _LABEL = np.array([[[2.0, 1.0, 2.0], [3.0, 2.0, 3.0]],
                     [[4.0, 3.0, 4.0], [5.0, 4.0, 5.0]]])

  def test_get_gradients_against_prediction(self):
    fake_model = _get_fake_model(self._IMAGE_1.shape, 3)

    grads, prediction = saliency_maps.get_gradients_against_prediction(
        tf.convert_to_tensor(self._IMAGE_1), fake_model)

    self.assertEqual(grads.shape, self._IMAGE_1.shape)
    self.assertEqual(prediction.shape, (2, 2, 3))

    grads, prediction = saliency_maps.get_gradients_against_prediction(
        tf.convert_to_tensor(self._IMAGE_1), fake_model)

    self.assertEqual(grads.shape, self._IMAGE_1.shape)
    self.assertEqual(prediction.shape, (2, 2, 3))

  def test_get_gradients_against_loss(self):
    fake_model = _get_fake_model(self._IMAGE_1.shape, 3)

    grads, prediction = saliency_maps.get_gradients_against_prediction(
        tf.convert_to_tensor(self._IMAGE_1), fake_model)

    self.assertEqual(grads.shape, self._IMAGE_1.shape)
    self.assertEqual(prediction.shape, (2, 2, 3))

    grads, prediction = saliency_maps.get_gradients_against_loss(
        tf.convert_to_tensor(self._IMAGE_1),
        fake_model,
        label=self._LABEL)

    self.assertEqual(grads.shape, self._IMAGE_1.shape)
    self.assertEqual(prediction.shape, (2, 2))

  def test_get_integrated_gradients(self):
    fake_model = _get_fake_model(self._IMAGE_1.shape, 3)

    igrads = saliency_maps.get_integrated_gradients(
        tf.convert_to_tensor(self._IMAGE_1, dtype=tf.float32),
        fake_model,
        gradient_base='prediction')

    self.assertEqual(igrads.shape, self._IMAGE_1.shape)

  def test_get_integrated_gradients_loss(self):
    fake_model = _get_fake_model(self._IMAGE_1.shape, 3)

    igrads = saliency_maps.get_integrated_gradients(
        tf.convert_to_tensor(self._IMAGE_1, dtype=tf.float32),
        fake_model,
        label=self._LABEL,
        gradient_base='loss')

    self.assertEqual(igrads.shape, self._IMAGE_1.shape)

  def test_get_blur_integrated_gradients(self):
    fake_model = _get_fake_model(self._IMAGE_1.shape, 3)

    igrads = saliency_maps.get_blur_integrated_gradients(
        tf.convert_to_tensor(self._IMAGE_1, dtype=tf.float32),
        fake_model,
        check_sanity=False)

    self.assertEqual(igrads.shape, self._IMAGE_1.shape)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
