r"""Rechunk the a Zarr dataset."""
import json
import logging

from absl import app
from absl import flags
import apache_beam as beam
import xarray
import xarray_beam


flags.DEFINE_string(
    'input_zarr', None, 'Input Zarr store.')
flags.mark_flag_as_required('input_zarr')
flags.DEFINE_string(
    'output_zarr', None, 'Output Zarr store.')
flags.mark_flag_as_required('output_zarr')
flags.DEFINE_string(
    'in_chunks', '{"time": 1024}',
    'Input chunking scheme, expressed in json format')
flags.DEFINE_string(
    'out_chunks', '{"time": 256}',
    'Output chunking scheme, expressed in json format')
flags.DEFINE_string('runner', None, 'beam.runners.Runner')

FLAGS = flags.FLAGS

# pylint: disable=expression-not-assigned
# pylint: disable=logging-format-interpolation


def main(argv):
  in_chunks = json.loads(FLAGS.in_chunks) if FLAGS.in_chunks else {}
  out_chunks = json.loads(FLAGS.out_chunks)
  in_ds = xarray.open_zarr(FLAGS.input_zarr, consolidated=True)
  for dim in set(in_ds.dims) - set(in_chunks):
    in_chunks[dim] = in_ds.chunks[dim]
  for dim in set(in_ds.dims) - set(out_chunks):
    out_chunks[dim] = -1
  in_ds = xarray.open_zarr(FLAGS.input_zarr, chunks=None, consolidated=True)

  logging.info('Input dataset:\n%s', in_ds)
  logging.info('Input chunking scheme: %s', in_chunks)
  logging.info('Output chunking scheme: %s', out_chunks)

  with beam.Pipeline(runner=FLAGS.runner, argv=argv) as p:
    (p
     # Note: splitting across variables in this dataset is a critical
     # optimization step here, because it allows rechunking to make use of much
     # larger intermediate chunks.
     | xarray_beam.DatasetToChunks(
         in_ds, in_chunks, num_threads=32, split_vars=True)
     | xarray_beam.Rechunk(in_ds.sizes, in_chunks, out_chunks, itemsize=8*4)
     | xarray_beam.ChunksToZarr(
         FLAGS.output_zarr, in_ds.chunk(), out_chunks, num_threads=32))


if __name__ == '__main__':
  app.run(main)
