r"""Processes zarr data to create training and validation zarr datasets.

Example Params for postprocessing zarr data:

DIR='gs://prj-leap-experimental'
input_zarr="$DIR/toy_2y_20210505-5.zarr" \
output_train_zarr="$DIR/zarr_train.zarr" \
output_val_zarr="$DIR/zarr_val.zarr" \
forecast_hr=6 \
variables 'Cloud mixing ratio, Geopotential Height, Relative humidity,
     Precipitable water, Surface pressure, Temperature, Total ozone, U component
     of wind, V component of wind' \

"""
import logging
from typing import Any, Dict, List, Tuple

import apache_beam as beam
from apache_beam.metrics import Metrics
import dateutil
import numpy as np
from PIL import Image
import xarray


def write_template(source_ds: xarray.Dataset,
                   times: xarray.DataArray,
                   zarr_store: str,
                   chunk_size: int = 1) -> None:
  ds = xarray.full_like(source_ds.chunk(), np.nan, dtype=np.float32)
  ds = ds.expand_dims({'t': times}, axis=(0))
  ds = ds.chunk({k: chunk_size for k in ds.dims if k not in ['i', 'j']})
  ds.to_zarr(zarr_store, compute=False, consolidated=True, mode='w')


def resize(arr: np.ndarray, width: int,
           height: int) -> np.ndarray:
  """Resizes the given array to the given width and height."""
  assert len(arr.shape) == 3
  num_channels = arr.shape[-1]
  imx = np.zeros((height, width, num_channels), dtype=np.float32)
  for iz in range(num_channels):
    imx[:, :,
        iz] = np.array(Image.fromarray(arr[:, :, iz]).resize((width, height)))

  return imx


def get_stacked_array(ds: xarray.Dataset, t: xarray.DataArray, fc: int,
                      features: List[str]) -> np.ndarray:
  """Fetches data for the given time, forecast and features."""

  def _get_feature_data(v: str) -> np.ndarray:
    d = ds[v].sel(t=t, fc=fc)
    return np.nan_to_num(d, np.nanmean(d))

  def _expand_array(x: np.ndarray) -> np.ndarray:
    if x.ndim < 3:
      return np.expand_dims(x, axis=2)
    return x

  results = [_get_feature_data(f) for f in features]
  return np.concatenate([_expand_array(result) for result in results], axis=-1)


def get_data(ds: xarray.Dataset, t: xarray.DataArray, fc: int,
             features: List[str], width: int,
             height: int) -> Tuple[np.ndarray, np.ndarray]:
  forecast = get_stacked_array(ds, t + fc * 3600, fc, features)
  analysis = get_stacked_array(ds, t + 2 * fc * 3600, 0, features)
  last_analysis = get_stacked_array(ds, t + fc * 3600, 0, features)
  last_forecast = get_stacked_array(ds, t, fc, features)
  last_error = last_analysis - last_forecast
  x_train = np.concatenate([forecast, last_error], axis=-1)
  y_train = analysis - forecast
  return resize(x_train, width, height), resize(y_train, width, height)


def convert_to_dataset(data: Tuple[np.ndarray, np.ndarray]) -> xarray.Dataset:
  inputs, output = data
  arrays = {}
  arrays['input'] = xarray.DataArray(inputs, dims=('i', 'j', 'z_input'))
  arrays['label'] = xarray.DataArray(output, dims=('i', 'j', 'z_output'))

  return xarray.Dataset(arrays)


class ZarrToXarray(beam.DoFn):
  """A DoFn that stacks and resizes a zarr array into an numpy array."""

  def __init__(self,
               split_name: str,
               root: str,
               output_store: str,
               forecast_hr: int,
               features: List[str],
               times: xarray.DataArray,
               width=512,
               height=256):
    self.split_name = split_name
    self.root = root
    self.forecast_hr = forecast_hr
    self.features = features
    self.width = width
    self.height = height
    self.output_store = output_store
    self.times = times

    try:
      self.z_root = xarray.open_zarr(root, consolidated=True, chunks=None)
    except FileNotFoundError:
      logging.error('file % is not found', root)

  def process(self, timestamp: xarray.DataArray):
    ds = convert_to_dataset(
        get_data(
            ds=self.z_root,
            t=timestamp,
            fc=self.forecast_hr,
            features=self.features,
            width=self.width,
            height=self.height))
    Metrics.counter(self.split_name, 'num_success').inc()
    ds = ds.expand_dims(('t'), axis=0)
    index = self.times.searchsorted(int(timestamp))
    region = {'t': slice(index, index + 1)}
    ds = ds.drop(list(ds.coords))
    ds.to_zarr(self.output_store, region=region, consolidated=True)
    Metrics.counter('zarr_to_xarray', 'num_success').inc()


def get_timestamp(t: str) -> int:
  """Parses time in a string format to timestamp."""
  time = dateutil.parser.parse(t)
  if time.tzinfo is None:
    raise ValueError(f'{t} time should contain tz info.')
  return time.timestamp()


def compute_metadata(ds: xarray.Dataset, start_time: int,
                     end_time: int) -> xarray.DataArray:
  return ds.t[np.logical_and(ds.t >= start_time, ds.t < end_time)]


def pipeline(input_zarr: str, output_train_zarr: str, output_val_zarr: str,
             forecast_hr: int, train_start_time: str, train_end_time: str,
             val_start_time: str, val_end_time: str, variables: List[str],
             resize_width: int, resize_height: int, chunk_size: int,
             pipeline_kwargs: Dict[str, Any]):
  """Executes beam pipeline to process zarr data into training and validation.

  Args:
    input_zarr: Path to Zarr volume to process.
    output_train_zarr: Path to output Zarr volume containing training data.
    output_val_zarr: Path to output Zarr volume containing validation data.
    forecast_hr: Forecast hour, e.g., 6.
    train_start_time: Start time for training data.
    train_end_time: End time for training data.
    val_start_time: Start time for training data.
    val_end_time: End time for validation data.
    variables: List of variables, e.g., Cloud mixing ratio, Geopotential Height.
    resize_width: Resize width, e.g., 512.
    resize_height: Resize height, e.g., 256.
    chunk_size: Chunk size of the ZARR output.
    pipeline_kwargs: Args to pass to the beam pipeline runner.
  """
  # Dimensions to compute stats over.
  in_ds = xarray.open_zarr(input_zarr, chunks=None, consolidated=True)

  # training dataset is split at 80% of the samples with shuffled indices
  train_times = compute_metadata(in_ds, get_timestamp(train_start_time),
                                 get_timestamp(train_end_time))
  # validation dataset is split at 20% of the samples with shuffled indices
  val_times = compute_metadata(in_ds, get_timestamp(val_start_time),
                               get_timestamp(val_end_time))
  sample_ds = convert_to_dataset(
      get_data(
          ds=in_ds,
          t=train_times.isel(t=0),
          fc=forecast_hr,
          features=variables,
          width=resize_width,
          height=resize_height))
  write_template(sample_ds, train_times, output_train_zarr,
                 chunk_size)
  write_template(sample_ds, val_times, output_val_zarr, chunk_size)

  splits = [['train', train_times, output_train_zarr],
            ['validation', val_times, output_val_zarr]]

  with beam.Pipeline(**pipeline_kwargs) as p:
    for split_name, times, output_store in splits:
      _ = (
          p
          | 'read_{}'.format(split_name) >> beam.Create(times)
          | 'zarr_to_xarray_{}'.format(split_name) >> beam.ParDo(
              ZarrToXarray(
                  split_name=split_name,
                  root=input_zarr,
                  output_store=output_store,
                  forecast_hr=forecast_hr,
                  features=variables,
                  width=resize_width,
                  height=resize_height,
                  times=times)))
